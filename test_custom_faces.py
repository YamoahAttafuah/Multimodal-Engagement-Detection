import os, glob, argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import cv2
import numpy as np

# ---- label presets (match training) ----
LABEL_PRESETS = {
    "7": {
        "names": ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"],
        "suffix":"7cls",
    },
    "5": {
        "names": ["Happy","Sad","Neutral","Angry","Surprise"],
        "suffix":"5cls",
    },
    "4": {
        "names": ["Positive","Negative","Neutral","Surprise"],
        "suffix":"4cls",
    },
    # 2-class Positive/Negative (your main setup)
    "2": {
        "names": ["Positive","Negative"],
        "suffix":"2cls",
    },
}


def crop_face_pil(pil_img):
    """Detect largest face and return a padded square crop as PIL image.
       If no face found, returns the original image."""
    img = np.array(pil_img)  # RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        img_bgr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)  # a bit smaller so it picks up your face
    )

    if len(faces) == 0:
        return pil_img  # fallback: no face found, use whole image

    # take largest face
    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    cx = x + fw // 2
    cy = y + fh // 2
    side = int(max(fw, fh) * 1.5)  # pad a bit around the face

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, cx + side // 2)
    y2 = min(h, cy + side // 2)

    face_crop = img_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def infer_expected_label(folder: str, EMOTIONS):
    """
    Try to guess the expected class name from the folder name.
    Works for 7/5/4/2-class setups.
    """
    f = folder.lower()

    # Generic alias map: folder_name_lower -> canonical label
    alias = {
        "happy": "Happy",
        "sad": "Sad",
        "neutral": "Neutral",
        "angry": "Angry",
        "anger": "Angry",
        "surprise": "Surprise",
        "surprised": "Surprise",
        "disgust": "Disgust",
        "fear": "Fear",

        "positive": "Positive",
        "pos": "Positive",
        "negative": "Negative",
        "neg": "Negative",
    }

    # First try aliases, but only if the target label exists in EMOTIONS
    if f in alias and alias[f] in EMOTIONS:
        return alias[f]

    # Then try exact match to any EMOTIONS (case-insensitive)
    for name in EMOTIONS:
        if f == name.lower():
            return name

    # Fallback: nothing recognized
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom_root", default="custom_data",
                    help="root folder with one subfolder per class "
                         "(e.g. positive/, negative/ or happy/, sad/, ...)")
    ap.add_argument("--ckpt", required=True,
                    help="path to trained checkpoint (.pth)")
    ap.add_argument("--labels", choices=["7","5","4","2"], default="2",
                    help="must match what you trained with")
    ap.add_argument("--arch", choices=["resnet50","resnet101"], default="resnet101")
    ap.add_argument("--debug_crops", default="debug_crops",
                    help="where to save cropped images")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- labels -----
    preset = LABEL_PRESETS[args.labels]
    EMOTIONS = preset["names"]
    NUM_CLASSES = len(EMOTIONS)

    print("Classes:", EMOTIONS)

    # ----- model -----
    if args.arch == "resnet101":
        m = models.resnet101(weights=None)
    else:
        m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)

    sd = torch.load(args.ckpt, map_location=device)
    m.load_state_dict(sd, strict=True)
    m.to(device)
    m.eval()

    # ---- transforms: crop (after face detection) then normalize ----
    tf_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    tf_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    os.makedirs(args.debug_crops, exist_ok=True)

    # Walk subfolders in custom_root; each subfolder = one "class"
    for folder in sorted(os.listdir(args.custom_root)):
        full_dir = os.path.join(args.custom_root, folder)
        if not os.path.isdir(full_dir):
            continue

        expected_name = infer_expected_label(folder, EMOTIONS)
        expected_idx = EMOTIONS.index(expected_name) if expected_name in EMOTIONS else None

        print(f"\n=== Folder: {folder} (expected: {expected_name}) ===")
        if expected_name is None:
            print("  [warn] Could not infer expected class from folder name; "
                  "will still print predictions but not accuracy.")

        total = 0
        correct = 0

        for img_path in sorted(glob.glob(os.path.join(full_dir, "*.*"))):
            try:
                img = Image.open(img_path).convert("RGB")

                # 1) detect + crop face
                face_img = crop_face_pil(img)

                # 2) resize + center crop to 224x224
                cropped = tf_crop(face_img)

                # 3) save debug crop (what model *actually* sees)
                debug_name = f"{folder}__{os.path.basename(img_path)}"
                cropped.save(os.path.join(args.debug_crops, debug_name))

                # 4) normalize + run model
                x = tf_norm(cropped).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = m(x)
                    pred_idx = int(logits.argmax(dim=1).item())
                pred_name = EMOTIONS[pred_idx]

            except Exception as e:
                print(f"  [skip] {img_path}: {e}")
                continue

            total += 1
            if expected_idx is not None and pred_idx == expected_idx:
                correct += 1

            print(f"  {os.path.basename(img_path)} -> {pred_name}")

        if expected_idx is not None and total > 0:
            print(f"  Accuracy in {folder}: {correct}/{total} = {correct/total:.3f}")

if __name__ == "__main__":
    main()

