# test_dfew_16f.py

import os
import glob
import csv
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


EMOTIONS = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
NUM_CLASSES = len(EMOTIONS)


def parse_label(tok: str) -> int:
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1
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
    """
    Test set: use all frames, one sample per frame.
    """
    def __init__(self, frames_root, split_items, transform):
        self.transform = transform
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

    def __getitem__(self, idx):
        cid, lab, path = self.clips[idx]
        img = Image.open(path).convert("RGB")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="dfew/DFEW-part2",
        help="path to DFEW-part2 (relative to project root)",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint path (defaults to pretrained_weight/best_resnet101_2cls_fold{fold}_16f_VAL.pth)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent      # dfew_emotion_code/
    project_root = script_dir.parent                  # AffectFusion_Project/

    args.root = str((project_root / args.root).resolve())

    if args.ckpt is None:
        args.ckpt = str(
            (script_dir / "pretrained_weight" / f"best_resnet101_2cls_fold{args.fold}_16f_VAL.pth").resolve()
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_root = os.path.join(args.root, "Clip", "clip_224x224_16f")
    splits_root = os.path.join(args.root, "EmoLabel_DataSplit")

    test_file = find_split(splits_root, args.fold, "test")
    test_items = load_split(test_file)

    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    ds_te = DFEWFramesTest(frames_root, test_items, tf_eval)
    dl_te = DataLoader(
        ds_te,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ResNet-101 backbone (same as training)
    model = models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    clip_logits = defaultdict(lambda: None)
    clip_lab = {}

    use_amp_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for x, y, cid in dl_te:
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

    war, uar = metrics_war_uar(y_true, y_pred, k_classes=NUM_CLASSES)
    print(f"[TEST] WAR {war:.4f}  UAR {uar:.4f}")


if __name__ == "__main__":
    main()

