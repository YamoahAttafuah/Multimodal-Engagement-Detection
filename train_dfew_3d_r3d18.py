import os, glob, csv, random, argparse
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.model_selection import train_test_split
from pathlib import Path

# ----- Label presets and collapse maps -----
# Original indices (0..6) = Happy, Sad, Neutral, Angry, Surprise, Disgust, Fear
LABEL_PRESETS = {
    # Original 7
    "7": {
        "names": ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"],
        "map":   None,  # no collapse
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
    if t in lut: return lut[t]
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
                if first in {"video_name","video","video_id","clip","id","name"} and last in {"label","emotion","class"}:
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

# ---------------- Stratified train/val split (from TRAIN only) ----------------
def stratified_split(items, val_frac=0.1, seed=42):
    """
    items: list of (clip_id, label after collapse)
    Returns: (train_items, val_items) stratified by label using sklearn.
    Falls back to a simple per-class split if a class is too tiny.
    """
    if not items:
        return [], []

    cids = [cid for cid, _ in items]
    labs = [lab for _, lab in items]

    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            cids, labs,
            test_size=val_frac,
            random_state=seed,
            stratify=labs
        )
        train_items = list(zip(X_tr, y_tr))
        val_items   = list(zip(X_va, y_va))
        return train_items, val_items
    except ValueError:
        # Fallback for ultra-tiny classes
        rng = random.Random(seed)
        by_lab = defaultdict(list)
        for cid, lab in items:
            by_lab[lab].append((cid, lab))
        tr, va = [], []
        for lab, lst in by_lab.items():
            rng.shuffle(lst)
            k = max(1, int(round(len(lst) * val_frac))) if len(lst) > 1 else 0
            va.extend(lst[:k]); tr.extend(lst[k:])
        rng.shuffle(tr); rng.shuffle(va)
        return tr, va

# ---------------- 3D Dataset: returns clips (C, T, H, W) ----------------
class DFEWClips3D(Dataset):
    """
    Each item is a whole clip:
      x: (C, T, H, W)  (e.g. 3x16x224x224)
      y: label
      cid: clip id
    """
    def __init__(self, frames_root, split_items, tfm_per_frame, num_frames=16):
        self.tfm = tfm_per_frame
        self.num_frames = num_frames
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
                self.clips.append((os.path.basename(d), lab, ims))
            else:
                missing.append(cid)
        if missing:
            print(f"[warn] missing {len(missing)} clips (e.g., {missing[:3]})")
        if not self.clips:
            raise RuntimeError("No frames found. Check --root and the 16f folder.")

    def __len__(self):
        return len(self.clips)

    def _select_frames(self, ims):
        """Select exactly self.num_frames paths from ims (uniform or pad last)."""
        if len(ims) >= self.num_frames:
            idxs = np.linspace(0, len(ims)-1, self.num_frames).astype(int)
            return [ims[i] for i in idxs]
        else:
            # pad by repeating last frame
            return ims + [ims[-1]] * (self.num_frames - len(ims))

    def __getitem__(self, i):
        cid, lab, ims = self.clips[i]
        chosen = self._select_frames(ims)
        frames = []
        for p in chosen:
            img = Image.open(p).convert("RGB")
            frames.append(self.tfm(img))  # (C,H,W)
        # frames: list of T tensors (C,H,W)
        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        return clip, lab, cid

# ---------------- Metrics ----------------
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

def eval_loader(model, dloader, device, K):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, _ in dloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda" if device.type=="cuda" else "cpu"):
                logits = model(x)  # (B,K)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return metrics_war_uar(y_true, y_pred, K=K)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dfew/DFEW-part2", help="path to DFEW-part2 (repo-relative default)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8, help="3D is heavy: start with 4–8")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--frames", type=int, default=16,
                    help="number of frames per clip (DFEW uses 16)")
    ap.add_argument("--val_frac", type=float, default=0.1, help="fraction of TRAIN used as validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--labels",
        choices=["7","5","4","2"],
        default="2",
        help="label preset: 7=original, 5=merged, 4=Pos/Neg/Neu/Sup, 2=Pos/Neg"
    )
    ap.add_argument("--out", type=str, default="models", help="folder to save checkpoints")
    args = ap.parse_args()

    # Resolve paths
    base = Path(__file__).resolve().parent.parent
    args.root = str((base / args.root).resolve())
    args.out  = str((base / args.out).resolve())
    os.makedirs(args.out, exist_ok=True)

    # Preset
    preset = LABEL_PRESETS[args.labels]
    EMOTIONS = preset["names"]
    NUM_CLASSES = len(EMOTIONS)
    collapse_map = preset["map"]
    suffix = preset["suffix"]

    print("Using labels preset:", args.labels, "->", EMOTIONS)

    # Seeds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Paths
    frames_root = os.path.join(args.root, "Clip", "clip_224x224_16f")
    splits_root = os.path.join(args.root, "EmoLabel_DataSplit")

    # Load TRAIN only, then make (train,val)
    train_file = find_split(splits_root, args.fold, "train")
    train_all  = load_split(train_file, collapse_map)
    print("Train split counts (after collapse):", Counter(lab for _, lab in train_all))
    train_items, val_items = stratified_split(train_all, val_frac=args.val_frac, seed=args.seed)

    # Video normalization (Kinetics-style)
    VIDEO_MEAN = [0.43216, 0.394666, 0.37645]
    VIDEO_STD  = [0.22803, 0.22145, 0.216989]

    # Per-frame transforms (applied to each frame)
    tf_train_frame = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(VIDEO_MEAN, VIDEO_STD),
    ])
    tf_eval_frame = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(VIDEO_MEAN, VIDEO_STD),
    ])

    # Datasets / Loaders
    ds_tr = DFEWClips3D(frames_root, train_items, tf_train_frame, num_frames=args.frames)
    ds_va = DFEWClips3D(frames_root, val_items,   tf_eval_frame,  num_frames=args.frames)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    # 3D Model (r3d_18 pretrained on Kinetics)
    weights = R3D_18_Weights.KINETICS400_V1
    m = r3d_18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.to(device)

    opt    = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda" if device.type=="cuda" else "cpu")
    torch.backends.cudnn.benchmark = True

    best_uar = -1.0
    best_path = os.path.join(args.out, f"best_r3d18_{suffix}_fold{args.fold}_{args.frames}f.pth")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        m.train(); running = 0.0; n = 0
        for x, y, _ in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if device.type=="cuda" else "cpu"):
                logits = m(x)            # x: (B, C, T, H, W)
                loss = nn.CrossEntropyLoss()(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0); n += x.size(0)

        # ---- Validate on VAL (clip-level) ----
        war, uar = eval_loader(m, dl_va, device, K=NUM_CLASSES)
        print(f"epoch {epoch:02d}  train_loss {running/max(n,1):.4f}  VAL_WAR {war:.4f}  VAL_UAR {uar:.4f}")

        if uar > best_uar:
            best_uar = uar
            torch.save(m.state_dict(), best_path)
            print(f"  ✅ new best VAL_UAR {best_uar:.4f} -> saved {best_path}")

    print(f"done. best VAL UAR={best_uar:.4f}")
    if os.path.isfile(best_path):
        print(f"best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
