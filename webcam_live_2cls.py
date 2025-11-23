import argparse
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# For your 2-class model
EMOTIONS = ["Positive", "Negative"]
NUM_CLASSES = len(EMOTIONS)

def crop_face_pil(pil_img):
    """
    Detect largest face and return a padded square crop as PIL image.
    If no face found, returns the original image.
    """
    import cv2
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
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return pil_img  # fallback

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to best_resnet101_2cls_fold1_16f_VAL.pth")
    ap.add_argument("--arch", choices=["resnet50","resnet101"], default="resnet101")
    ap.add_argument("--camera", type=int, default=0,
                    help="webcam index (default 0)")
    ap.add_argument("--window", type=int, default=16,
                    help="number of frames to average over")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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

    # ----- transforms -----
    tf_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    tf_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ----- webcam -----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Could not open webcam", args.camera)
        return

    print("✅ Webcam opened. Press 'q' to quit.")

    # store last N frame logits
    from collections import deque
    logit_window = deque(maxlen=args.window)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # BGR -> RGB -> PIL
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # face crop
            face_img = crop_face_pil(pil_img)

            # resize + center crop
            cropped = tf_crop(face_img)

            # to tensor + normalize
            x = tf_norm(cropped).unsqueeze(0).to(device)

            # forward
            with torch.amp.autocast("cuda" if device.type=="cuda" else "cpu"):
                logits = m(x)  # (1,2)

            logit_window.append(logits.squeeze(0).cpu())

            # average over window
            avg_logits = torch.stack(list(logit_window), dim=0).mean(0)
            pred_idx = int(avg_logits.argmax().item())
            pred_name = EMOTIONS[pred_idx]

            # simple confidence (softmax)
            probs = torch.softmax(avg_logits, dim=0)
            conf = float(probs[pred_idx].item())

            # overlay on frame
            text = f"{pred_name} ({conf*100:.1f}%)"
            color = (0,255,0) if pred_name == "Positive" else (0,0,255)
            cv2.putText(frame, text, (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            cv2.imshow("Webcam Emotion (q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
