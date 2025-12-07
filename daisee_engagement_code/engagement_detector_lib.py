import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path

# -------------------------------------------------------------------
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

SEQ_LENGTH = 75
MODEL_PATH = BASE_DIR / "pretrained_weight" / "rf_daisee_model_GAMMA_53features.h5"
JSON_PATH = BASE_DIR / "selected_features_53.json"

# Globals
MODEL = None
INDICES = None
FACE_MESH = None

# Thresholds
DEFAULT_THRESHOLDS = {
    "Boredom": 0.32,
    "Engagement": 0.65,
    "Confusion": 0.21,
    "Frustration": 0.17,
}


def initialize():
    """
    Lazily load:
    - selected feature indices (JSON_PATH)
    - DAiSEE engagement model (MODEL_PATH)
    - MediaPipe FaceMesh
    """
    global MODEL, INDICES, FACE_MESH

    if MODEL is not None:
        return

    # ----- Load selected feature indices -----
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {JSON_PATH}")
    with JSON_PATH.open("r") as f:
        INDICES = np.array(json.load(f))

    # ----- Load model -----
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    try:
        MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {MODEL_PATH}: {e}")

    # ----- Init MediaPipe FaceMesh -----
    FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def process_frame(frame):
    """
    Extract 53 features from a single frame using MediaPipe landmarks.
    Returns a (53,) feature vector or None if no face is detected.
    """
    if FACE_MESH is None:
        return None
    if frame is None:
        return None

    # Ensure RGB
    if frame.shape[-1] == 3:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = frame

    results = FACE_MESH.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    # Basic normalization: center on nose, normalize by inter-eye distance
    nose = coords[1]
    left_eye = coords[33]
    right_eye = coords[263]

    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist < 0.0001:
        eye_dist = 1.0

    centered = coords - nose
    normalized = centered / eye_dist

    try:
        return normalized.flatten()[INDICES]
    except IndexError:
        # In case INDICES goes out of bounds
        return None


def decide_label(preds, thresholds):
    """
    preds: array-like of shape (4,) -> [Boredom, Confusion, Engagement, Frustration]
    thresholds: dict mapping label -> threshold
    """
    scores = {
        "Boredom": preds[0],
        "Confusion": preds[1],
        "Engagement": preds[2],
        "Frustration": preds[3],
    }

    deviations = []
    for label, score in scores.items():
        thresh = thresholds.get(label, 0.5)
        deviation = score - thresh
        if deviation > 0:
            deviations.append({"label": label, "deviation": deviation})

    if not deviations:
        return "Neutral"

    # Pick label that exceeded its threshold by the largest margin
    deviations.sort(key=lambda x: x["deviation"], reverse=True)
    return deviations[0]["label"]


def get_engagement_label(frames, thresholds=None):
    """
    frames: numpy array or list of frames (H, W, 3) in BGR.
            We use at most the last 75 frames and pad if < 75.
    thresholds: optional dict to override DEFAULT_THRESHOLDS.

    Returns one of:
        "Engagement", "Boredom", "Confusion", "Frustration", "Neutral"
    """
    # Load model + indices + FaceMesh if not already loaded
    initialize()

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Use only the last SEQ_LENGTH frames
    current_frames = frames[-SEQ_LENGTH:]

    processed_seq = []
    for frame in current_frames:
        feats = process_frame(frame)
        if feats is not None:
            processed_seq.append(feats)

    # No valid features -> Neutral
    if not processed_seq:
        return "Neutral"

    # Pad at the beginning if we have fewer than SEQ_LENGTH
    while len(processed_seq) < SEQ_LENGTH:
        processed_seq.insert(0, processed_seq[0])

    # Run inference
    input_tensor = np.array([processed_seq])  # shape (1, 75, 53)
    predictions = MODEL.predict(input_tensor, verbose=0)[0]

    return decide_label(predictions, thresholds)

