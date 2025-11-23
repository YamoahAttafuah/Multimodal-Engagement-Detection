import os, glob, csv, argparse
from collections import Counter
from pathlib import Path

# ---------- Base 7-class setup ----------
# 0=Happy, 1=Sad, 2=Neutral, 3=Angry, 4=Surprise, 5=Disgust, 6=Fear
BASE_EMOTIONS = ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"]
BASE_NUM_CLASSES = len(BASE_EMOTIONS)

def parse_label(tok: str) -> int:
    """Parse label from DFEW split file into 0..6 index."""
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1
        if not (0 <= idx < BASE_NUM_CLASSES):
            raise ValueError(f"Label out of range: {tok}")
        return idx
    t = t.lower()
    alias = {"happiness":"happy","surprised":"surprise",
             "neutrality":"neutral","anger":"angry"}
    t = alias.get(t, t)
    lut = {e.lower(): i for i, e in enumerate(BASE_EMOTIONS)}
    if t in lut:
        return lut[t]
    raise ValueError(f"Unrecognized label token: {tok}")

def load_split_raw(path: str):
    """Return list of (clip_id, label_idx 0..6) with NO collapsing."""
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
                if first in {"video_name","video","video_id","clip","id","name"} and \
                   last  in {"label","emotion","class"}:
                    continue
                clip = str(row[0]).strip()
                lab  = parse_label(str(row[-1]))
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
    """Find split file in EmoLabel_DataSplit/<kind>(single-labeled)/."""
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

# ---------- Mapping schemes ----------
def apply_map(labels, mapping):
    if mapping is None:
        return labels
    return [mapping[l] for l in labels]

SCHEMES = {
    "7_orig": {
        "names": BASE_EMOTIONS,
        "map": None,
    },
    # your 4-class mapping (current)
    # Positive: Happy
    # Negative: Sad, Angry, Disgust, Fear
    # Neutral : Neutral
    # Surprise: Surprise
    "4_pos_neg_neu_sup": {
        "names": ["Positive","Negative","Neutral","Surprise"],
        "map":   {0:0, 1:1, 2:2, 3:1, 4:3, 5:1, 6:1},
    },
    # 2-class variant you suggested:
    # Positive: Happy + Neutral + Surprise
    # Negative: Sad + Angry + Disgust + Fear
    "2_pos_neg": {
        "names": ["Positive","Negative"],
        "map":   {0:0, 1:1, 2:0, 3:1, 4:0, 5:1, 6:1},
    },
    # (optional) 5-class mapping from before
    "5_merge_disgust_fear": {
        "names": ["Happy","Sad","Neutral","Angry","Surprise"],
        "map":   {0:0, 1:1, 2:2, 3:3, 4:4, 5:3, 6:1},
    },
}

def print_counts(name, labels, scheme):
    names = scheme["names"]
    mapped = apply_map(labels, scheme["map"])
    cnt = Counter(mapped)
    total = len(mapped)
    print(f"  {name}:")
    for idx, cname in enumerate(names):
        print(f"    {cname:9s}: {cnt.get(idx,0):5d}")
    print(f"    TOTAL   : {total:5d}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dfew/DFEW-part2",
                    help="path to DFEW-part2 (repo-relative default)")
    ap.add_argument("--folds", type=int, nargs="*", default=[1,2,3,4,5],
                    help="which folds to analyze (default: 1 2 3 4 5)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent
    root = str((base / args.root).resolve())
    splits_root = os.path.join(root, "EmoLabel_DataSplit")

    print("DFEW root:", root)
    print("Folds   :", args.folds)

    for fold in args.folds:
        print("\n" + "="*60)
        print(f"FOLD {fold}")
        print("="*60)

        for split_kind in ["train", "test"]:
            split_file = find_split(splits_root, fold, split_kind)
            items = load_split_raw(split_file)
            labels = [lab for _, lab in items]

            print(f"\n--- {split_kind.upper()} split ---")
            # 7-class original
            print_counts("7-class original", labels, SCHEMES["7_orig"])
            # 4-class mapping (Pos/Neg/Neutral/Surprise)
            print_counts("4-class Pos/Neg/Neu/Sup", labels, SCHEMES["4_pos_neg_neu_sup"])
            # 2-class mapping (Pos=Happy+Neutral+Surprise, Neg=others)
            print_counts("2-class Pos/Neg", labels, SCHEMES["2_pos_neg"])
            # optional 5-class you used earlier
            print_counts("5-class merged", labels, SCHEMES["5_merge_disgust_fear"])

if __name__ == "__main__":
    main()
