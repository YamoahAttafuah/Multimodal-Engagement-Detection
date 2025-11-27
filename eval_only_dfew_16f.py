import os, glob, csv, argparse
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path

# ----- Label presets and collapse maps -----
# Original base indices (0..6) = Happy, Sad, Neutral, Angry, Surprise, Disgust, Fear
LABEL_PRESETS = {
    # 7-class (no collapse)
    "7": {
        "names": ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"],
        "map":   None,
        "suffix":"7cls"
    },
    # 5-class: merge Disgust→Angry, Fear→Sad
    "5": {
        "names": ["Happy","Sad","Neutral","Angry","Surprise"],
        "map":   {0:0, 1:1, 2:2, 3:3, 4:4, 5:3, 6:1},
        "suffix":"5cls"
    },
    # 4-class: Positive / Negative / Neutral / Surprise
    # Positive: Happy
    # Negative: Sad, Angry, Disgust, Fear
    # Neutral : Neutral
    # Surprise: Surprise
    "4": {
        "names": ["Positive","Negative","Neutral","Surprise"],
        "map":   {0:0, 1:1, 2:2, 3:1, 4:3, 5:1, 6:1},
        "suffix":"4cls"
    },
    # 2-class: Positive / Negative
    # Positive: Happy, Neutral, Surprise
    # Negative: Sad, Angry, Disgust, Fear
    "2": {
        "names": ["Positive","Negative"],
        "map":   {0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:1},
        "suffix":"2cls"
    },
}

def collapse_label(idx: int, collapse_map):
    return idx if collapse_map is None else collapse_map[idx]

# ---------------- Base (original) label helpers ----------------
BASE_EMOTIONS = ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"]
BASE_NUM_CLASSES = len(BASE_EMOTIONS)

def parse_label(tok: str) -> int:
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1                # 1..7 -> 0..6
        if not (0 <= idx < BASE_NUM_CLASSES):
            raise ValueError(f"Label out of range: {tok}")
        return idx
    t = t.lower()
    alias = {"happiness":"happy","surprised":"surprise","neutrality":"neutral","anger":"angry"}
    t = alias.get(t, t)
    lut = {e.lower(): i for i, e in enumerate(BASE_EMOTIONS)}
    if t in lut:
        return lut[t]
    raise ValueError(f"Unrecognized label token: {tok}")

def load_split(path: str, collapse_map=None):
    """Reads CSV/TXT. For CSV, assumes first col=clip id, last col=label. Applies collapse_map."""
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
                lab  = parse_label(str(row[-1]))
                if clip:
                    items.append((clip, collapse_label(lab, collapse_map)))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", " ").split() if p]
                lab = parse_label(parts[-1])
                items.append((parts[0], collapse_label(lab, collapse_map)))
    if not items:
        raise RuntimeError(f"No items in {path}")
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

class DFEWFramesTest(Dataset):
    """Eval: per_clip=16 (use all frames)."""
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
                # store all frames for deterministic pass
                for p in ims:
                    self.clips.append((os.path.basename(d), lab, p))
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dfew/DFEW-part2", help="path to DFEW-part2 (repo-relative default)")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models_dir", type=str, default="models",
                    help="folder with checkpoints (relative to repo root)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="if None: models/best_resnet101_<suffix>_fold{fold}_16f_VAL.pth")
    ap.add_argument(
        "--labels",
        choices=["7","5","4","2"],
        default="2",
        help="label preset: 7, 5, 4, or 2 (must match training)"
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent

    # Resolve dataset + models under the repo parent (same as training)
    args.root = str((base / args.root).resolve())
    args.models_dir = str((base / args.models_dir).resolve())

    # ALSO resolve --ckpt relative to the same base if it's a relative path
    if args.ckpt is not None and not os.path.isabs(args.ckpt):
        args.ckpt = str((base / args.ckpt).resolve())


    # Preset (same as training)
    preset = LABEL_PRESETS[args.labels]
    EMOTIONS = preset["names"]
    NUM_CLASSES = len(EMOTIONS)
    collapse_map = preset["map"]
    suffix = preset["suffix"]

    print("Eval with labels preset:", args.labels, "->", EMOTIONS)

    if args.ckpt is None:
        from os.path import join
        args.ckpt = join(args.models_dir, f"best_resnet101_{suffix}_fold{args.fold}_16f_VAL.pth")

    print("Checkpoint:", args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_root = os.path.join(args.root, "Clip", "clip_224x224_16f")
    splits_root = os.path.join(args.root, "EmoLabel_DataSplit")
    test_file   = find_split(splits_root, args.fold, "test")
    test_items  = load_split(test_file, collapse_map)
    print("Test split counts (after collapse):", Counter(lab for _, lab in test_items))

    tf_eval = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds_te = DFEWFramesTest(frames_root, test_items, tf_eval)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    # Model
    m = models.resnet101(weights=None)  # for eval we just load your trained weights
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    sd = torch.load(args.ckpt, map_location=device)
    m.load_state_dict(sd, strict=True)
    m.to(device)
    m.eval()

    # Aggregate per clip
    clip_logits = defaultdict(lambda: None); clip_lab = {}
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
    print(f"[TEST] classes={EMOTIONS}")
    print(f"[TEST] WAR {war:.4f}  UAR {uar:.4f}")

if __name__ == "__main__":
    main()
