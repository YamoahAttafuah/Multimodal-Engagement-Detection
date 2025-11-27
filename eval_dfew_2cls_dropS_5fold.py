import os, glob, csv, argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------- Base labels from DFEW ----------
# 0 = Happy, 1 = Sad, 2 = Neutral, 3 = Angry, 4 = Surprise, 5 = Disgust, 6 = Fear
BASE_EMOTIONS = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
BASE_NUM_CLASSES = len(BASE_EMOTIONS)

# 2-class mapping (Positive / Negative)
# Positive: Happy, Neutral
# Negative: Sad, Angry, Disgust, Fear
# BUT we DROP Surprise (index 4) completely before mapping.
EMOTIONS = ["Positive", "Negative"]
NUM_CLASSES = len(EMOTIONS)

two_class_map = {
    0: 0,  # Happy    -> Positive
    1: 1,  # Sad      -> Negative
    2: 0,  # Neutral  -> Positive
    3: 1,  # Angry    -> Negative
    4: 0,  # Surprise -> (would be Positive, but we drop all 4 anyway)
    5: 1,  # Disgust  -> Negative
    6: 1,  # Fear     -> Negative
}

def collapse_label(idx: int) -> int:
    return two_class_map[idx]

# ---------- label parsing (same as training) ----------

def parse_label(tok: str) -> int:
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1  # 1..7 -> 0..6
        if not (0 <= idx < BASE_NUM_CLASSES):
            raise ValueError(f"Label out of range: {tok}")
        return idx
    t = t.lower()
    alias = {
        "happiness": "happy",
        "surprised": "surprise",
        "neutrality": "neutral",
        "anger": "angry",
    }
    t = alias.get(t, t)
    lut = {e.lower(): i for i, e in enumerate(BASE_EMOTIONS)}
    if t in lut:
        return lut[t]
    raise ValueError(f"Unrecognized label token: {tok}")

# ---------- load one split file, drop Surprise, map to 2 classes ----------

def load_split(path: str):
    """
    Reads CSV/TXT. For CSV, assumes first col=clip id, last col=label.
    - Drops all Surprise clips (base label index 4).
    - Maps remaining labels to 2-class Positive/Negative.
    """
    items = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                first = str(row[0]).strip().lower()
                last  = str(row[-1]).strip().lower()
                # Skip header like: video_name,label
                if first in {"video_name","video","video_id","clip","id","name"} and \
                   last  in {"label","emotion","class"}:
                    continue
                clip = str(row[0]).strip()
                if not clip:
                    continue
                base_idx = parse_label(str(row[-1]))
                if base_idx == 4:     # Surprise -> drop
                    continue
                lab = collapse_label(base_idx)
                items.append((clip, lab))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", " ").split() if p]
                base_idx = parse_label(parts[-1])
                if base_idx == 4:
                    continue
                lab = collapse_label(base_idx)
                items.append((parts[0], lab))

    if not items:
        raise RuntimeError(f"No items in {path} (maybe everything got filtered?)")
    return items

def find_split(base_dir, fold, split_kind):
    d = os.path.join(base_dir, f"{split_kind}(single-labeled)")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"missing dir: {d}")
    cand = sorted(
        glob.glob(os.path.join(d, f"*{fold}*.csv")) +
        glob.glob(os.path.join(d, f"*{fold}*.txt"))
    )
    if not cand:
        raise FileNotFoundError(f"no split file for fold {fold} in {d}")
    return cand[0]

# ---------- Dataset for test (all 16 frames per clip) ----------

class DFEWFramesTest(Dataset):
    """
    Eval: use ALL frames from the 16f folder.
    We expand each clip into multiple items (frame, label, clip_id).
    """
    def __init__(self, frames_root, split_items, tfm):
        self.tfm = tfm
        self.clips = []
        missing = []
        for cid, lab in split_items:
            d = os.path.join(frames_root, str(cid))
            if not os.path.isdir(d):
                d2 = os.path.join(frames_root, str(cid).zfill(5))
                d = d2 if os.path.isdir(d2) else d
            ims = sorted(
                glob.glob(os.path.join(d, "*.jpg")) +
                glob.glob(os.path.join(d, "*.jpeg")) +
                glob.glob(os.path.join(d, "*.png"))
            )
            if ims:
                cid_base = os.path.basename(d)
                for p in ims:
                    self.clips.append((cid_base, lab, p))
            else:
                missing.append(cid)
        if missing:
            print(f"[warn] missing {len(missing)} clips (e.g., {missing[:3]})")
        if not self.clips:
            raise RuntimeError("No frames found. Check --root and the 16f folder.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        cid, lab, path = self.clips[i]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        return x, lab, cid

# ---------- Metrics ----------

def metrics_war_uar(y_true, y_pred, K):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    war = float((y_true == y_pred).mean())
    rec = []
    for k in range(K):
        m = (y_true == k)
        if m.sum():
            rec.append(float((y_pred[m] == k).mean()))
    uar = float(np.mean(rec)) if rec else 0.0
    return war, uar

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dfew/DFEW-part2",
                    help="path to DFEW-part2 (repo-relative default)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models_dir", type=str, default="models",
                    help="folder with checkpoints (relative to repo root)")
    args = ap.parse_args()

    # Resolve paths relative to repo parent (same idea as training)
    base = Path(__file__).resolve().parent.parent
    args.root = str((base / args.root).resolve())
    args.models_dir = str((base / args.models_dir).resolve())

    print(">>> 2-class DFEW TEST (Positive / Negative, Surprise dropped)")
    print("Classes:", EMOTIONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    frames_root = os.path.join(args.root, "Clip", "clip_224x224_16f")
    splits_root = os.path.join(args.root, "EmoLabel_DataSplit")

    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    folds = [1, 2, 3, 4, 5]
    fold_results = []

    for fold in folds:
        print("\n============================")
        print(f"   TESTING FOLD {fold}")
        print("============================")

        ckpt_path = os.path.join(
            args.models_dir, f"best_resnet101_2cls_fold{fold}_16f_VAL.pth"
        )
        if not os.path.isfile(ckpt_path):
            print(f"[fold {fold}] WARNING: checkpoint not found: {ckpt_path}")
            continue
        print(f"[fold {fold}] using checkpoint: {ckpt_path}")

        # ---- Load TEST split for this fold ----
        test_file = find_split(splits_root, fold, "test")
        test_items = load_split(test_file)
        print(f"[fold {fold}] Test split counts (2-class, Surprise dropped):",
              Counter(lab for _, lab in test_items))

        # ---- Dataset / loader ----
        ds_te = DFEWFramesTest(frames_root, test_items, tf_eval)
        dl_te = DataLoader(
            ds_te,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        # ---- Model ----
        m = models.resnet101(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        sd = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(sd, strict=True)
        m.to(device)
        m.eval()

        # ---- Aggregate per clip ----
        clip_logits = defaultdict(lambda: None)
        clip_lab = {}

        with torch.no_grad():
            for x, y, cid in dl_te:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda" if device.type=="cuda" else "cpu"):
                    logits = m(x)
                for i, c in enumerate(cid):
                    clip_lab[c] = int(y[i])
                    cur = logits[i].detach().cpu()
                    clip_logits[c] = cur if clip_logits[c] is None else (clip_logits[c] + cur)

        y_true, y_pred = [], []
        for c, logit_sum in clip_logits.items():
            y_true.append(clip_lab[c])
            y_pred.append(int(logit_sum.argmax().item()))

        war, uar = metrics_war_uar(y_true, y_pred, K=NUM_CLASSES)
        print(f"[fold {fold}] TEST WAR {war:.4f}  UAR {uar:.4f}")

        fold_results.append((fold, war, uar))

    # ------- Summary over folds -------
    if fold_results:
        print("\n======= TEST SUMMARY OVER 5 FOLDS (2-class, Surprise dropped) =======")
        mean_war = sum(w for _, w, _ in fold_results) / len(fold_results)
        mean_uar = sum(u for _, _, u in fold_results) / len(fold_results)
        for f, w, u in fold_results:
            print(f"Fold {f}: WAR={w:.4f}  UAR={u:.4f}")
        print(f"Mean WAR over folds: {mean_war:.4f}")
        print(f"Mean UAR over folds: {mean_uar:.4f}")
    else:
        print("No folds were evaluated (missing checkpoints?).")

if __name__ == "__main__":
    main()
