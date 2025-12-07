import cv2
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import models, transforms

# -------------------------------------------------------------------

# -------------------------------------------------------------------
_model = None
_device = None

_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_classes = ["Positive", "Negative"]


BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "pretrained_weight" / "best_resnet101_2cls_fold1_16f_VAL.pth"


def _choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def _init_model():
    """
    Lazy init: choose device, build ResNet101, load 2-class checkpoint
    from dfew_emotion_code/pretrained_weight/.
    """
    global _model, _device
    if _model is not None:
        return

    _device = _choose_device()
    print("[DFEW] using device:", _device)

    # Build bare ResNet-101
    m = models.resnet101(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)  # Positive / Negative

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"[DFEW] checkpoint not found: {CKPT_PATH}")

    print("[DFEW] loading checkpoint:", CKPT_PATH)
    sd = torch.load(str(CKPT_PATH), map_location=_device)

    m.load_state_dict(sd, strict=True)
    m.to(_device).eval()

    _model = m
    print("[DFEW] checkpoint loaded successfully.")


def predict_dfew_valence(face_bgr):
    """
    Input:
        face_bgr: cropped face, H x W x 3, BGR from OpenCV.

    Output:
        (label_str, conf_float, probs_list)

        label_str  = "Positive" or "Negative"
        conf_float = probability of that class (0..1)
        probs_list = [p_pos, p_neg]
    """
    _init_model()

    # Convert BGR (OpenCV) -> RGB (PIL)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    x = _tf(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    label = _classes[idx]
    conf = float(probs[idx])

    return label, conf, probs.tolist()

