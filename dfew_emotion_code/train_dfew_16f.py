# train_dfew_16f.py

import os
import glob
import csv
import random
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


# Labels: 1→Happy, 2→Sad, 3→Neutral, 4→Angry, 5→Surprise, 6→Disgust, 7→Fear
EMOTIONS = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTIONS)


def parse_label(tok: str) -> int:
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1  # 1..7 -> 0..6
        if not (0 <= idx < NUM_CLASSES):
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

    lut = {e.lower(): i for i, e in enumerate(EMOTIONS)}
    if t in lut:
        return lut[t]

    raise ValueError(f"Unrecognized label token: {tok}")


def load_split(path: str):
    """
    Read a CSV or TXT split file.

    CSV: first column = clip id, last column = label.
    TXT: first token = clip id, last token = label.
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
                last = str(row[-1]).strip().lower()

                if first in {"video_name", "video", "video_id", "clip", "id", "name"} \
                        and last in {"label", "emotion", "class"}:
                    continue

                clip = str(row[0]).strip()
                lab = parse_label(str(row[-1]))
                if clip:
                    items.append((clip, lab))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", " ").split() if p]
                items.append((parts[0], parse_label(parts[-1])))

    if not items:
        raise RuntimeError(f"No items in {path}")

    return items


def find_split(base_dir, fold, split_kind):
    """
    Find split file for given fold and split kind ('train' or 'test').
    """
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


def stratified_split(items, val_frac=0.1, seed=42):
    """
    items: list of (clip_id, label)

    Returns:
        train_items, val_items
    """
    if not items:
        return [], []

    cids = [cid for cid, _ in items]
    labs = [lab for _, lab in items]

    try:
        x_tr, x_va, y_tr, y_va = train_test_split(
            cids,
            labs,
            test_size=val_frac,
            random_state=seed,
            stratify=labs,
        )
        train_items = list(zip(x_tr, y_tr))
        val_items = list(zip(x_va, y_va))
        return train_items, val_items

    except ValueError:
        rng = random.Random(seed)
        by_lab = defaultdict(list)
        for cid, lab in items:
            by_lab[lab].append((cid, lab))

        tr, va = [], []
        for lab, lst in by_lab.items():
            rng.shuffle(lst)
            if len(lst) > 1:
                k = max(1, int(round(len(lst) * val_frac)))
            else:
                k = 0
            va.extend(lst[:k])
            tr.extend(lst[k:])

        rng.shuffle(tr)
        rng.shuffle(va)
        return tr, va


class DFEWFrames(Dataset):
    """
    Train: per_clip=1  (one random frame per clip, per epoch)
    Eval : per_clip=16 (cover all 16 frames)
    """
    def __init__(self, frames_root, split_items, transform, per_clip=1):
        self.transform = transform
        self.per_clip = per_clip
        self.clips = []

        missing = []
        for cid, lab in split_items:
            d = os.path.join(frames_root, str(cid))
            if not os.path.isdir(d):
                d2 = os.path.join(frames_root, str(cid).zfill(5))
                if os.path.isdir(d2):
                    d = d2

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
        return len(self.clips) * self.per_clip

    def __getitem__(self, idx):
        ci = idx // self.per_clip
        cid, lab, ims = self.clips[ci]
        frame_idx = random.randrange(len(ims))

        img = Image.open(ims[frame_idx]).convert("RGB")
        x = self.transform(img)
        return x, lab, cid


def metrics_war_uar(y_true, y_pred, k_classes=NUM_CLASSES):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    war = float((y_true == y_pred).mean())

    rec = []
    for k in range(k_classes):
        mask = (y_true == k)
        if mask.sum():
            rec.append(float((y_pred[mask] == k).mean()))

    uar = float(np.mean(rec)) if rec else 0.0
    return war, uar


def eval_loader(model, dloader, device):
    model.eval()
    clip_logits = defaultdict(lambda: None)
    clip_lab = {}

    use_amp_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for x, y, cid in dloader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(use_amp_device):
                logits = model(x)

            for i, c in enumerate(cid):
                clip_lab[c] = int(y[i])
                cur = logits[i].detach().cpu()
                clip_logits[c] = cur if clip_logits[c] is None else (clip_logits[c] + cur)

    y_true, y_pred = [], []
    for c, logit_sum in clip_logits.items():
        y_true.append(clip_lab[c])
        y_pred.append(int(logit_sum.argmax().item()))

    return metrics_war_uar(y_true, y_pred, k_classes=NUM_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="dfew/DFEW-part2",
        help="path to DFEW-part2 (relative to project root)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="pretrained_weight")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    args.root = str((project_root / args.root).resolve())
    args.out = str((script_dir / args.out).resolve())
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_root = os.path.join(args.root, "Clip", "clip_224x224_16f")
    splits_root = os.path.join(args.root, "EmoLabel_DataSplit")

    train_file = find_split(splits_root, args.fold, "train")
    train_all = load_split(train_file)
    print("Train split counts:", Counter(lab for _, lab in train_all))

    train_items, val_items = stratified_split(
        train_all,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    ds_tr = DFEWFrames(frames_root, train_items, tf_train, per_clip=1)
    ds_va = DFEWFrames(frames_root, val_items, tf_eval, per_clip=16)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ResNet-101 backbone
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True

    best_uar = -1.0
    best_path = os.path.join(
        args.out,
        f"best_resnet101_7cls_fold{args.fold}_16f_VAL.pth",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y, _ in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            use_amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(use_amp_device):
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        war, uar = eval_loader(model, dl_va, device)
        avg_loss = running_loss / max(n_samples, 1)
        print(
            f"epoch {epoch:02d}  "
            f"train_loss {avg_loss:.4f}  "
            f"VAL_WAR {war:.4f}  VAL_UAR {uar:.4f}"
        )

        if uar > best_uar:
            best_uar = uar
            torch.save(model.state_dict(), best_path)
            print(f"  new best VAL_UAR {best_uar:.4f} -> saved {best_path}")

    print(f"done. best VAL UAR = {best_uar:.4f}")
    if os.path.isfile(best_path):
        print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

