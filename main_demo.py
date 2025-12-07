# main_demo.py

import os
import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import tensorflow as tf

from crop_utils import crop_face

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "daisee_engagement_code"))
sys.path.append(os.path.join(BASE_DIR, "dfew_emotion_code"))

from engagement_detector_lib import get_engagement_label, DEFAULT_THRESHOLDS
from inference_dfew_2cls import predict_dfew_valence


# Configuration
CAM_WIDTH = 640
CAM_HEIGHT = 360

FACE_CONF_THRESH = 0.50
FUSION_MARGIN = 0.10

AUDIO_SR = 22050
AUDIO_CHUNK_DUR = 5.0
AUDIO_SILENCE_THRESH = 0.001

VAL_HISTORY_LENGTH = 20
ENG_BUFFER_LENGTH = 75


def run_valence(frame_bgr):
    """
    Run DFEW 2-class model on a single frame.

    Returns:
        (label, conf), label ∈ {"Positive", "Negative", "Uncertain"}
    """
    face = crop_face(frame_bgr)

    if face is None:
        print("[DFEW] Emotion: Uncertain (no face)")
        return "Uncertain", 0.0

    raw_label, conf, _ = predict_dfew_valence(face)

    if conf < FACE_CONF_THRESH:
        print(f"[DFEW] Emotion: Uncertain ({raw_label} {conf*100:.1f}%)")
        return "Uncertain", conf

    print(f"[DFEW] Emotion: {raw_label} {conf*100:.1f}%")
    return raw_label, conf


def start_engagement_worker(frame_buffer, thresholds):
    """
    Background thread for DAiSEE engagement.
    """
    state = {
        "last_eng_label": "Neutral",
        "stop": False,
    }

    def worker():
        time.sleep(1.0)

        while not state["stop"]:
            if len(frame_buffer) > 0:
                frames = list(frame_buffer)
                try:
                    label = get_engagement_label(frames, thresholds)
                    state["last_eng_label"] = label
                    print("[ENG ]", label)
                except Exception as e:
                    print("[ENG ] error:", e)
            time.sleep(0.5)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return state


_audio_model = None


def _load_audio_model():
    global _audio_model
    if _audio_model is not None:
        return _audio_model

    script_dir = Path(__file__).resolve().parent
    weights_dir = script_dir / "ravdess_audio_code" / "pretrained_weight"

    keras_files = list(weights_dir.glob("*.keras"))
    if not keras_files:
        raise FileNotFoundError(f"[AUDIO] No .keras model found in {weights_dir}")
    model_path = keras_files[0]

    print("[AUDIO] loading model from:", model_path)
    _audio_model = tf.keras.models.load_model(str(model_path))
    return _audio_model


def _record_audio(dur=AUDIO_CHUNK_DUR, sr=AUDIO_SR):
    audio = sd.rec(
        int(dur * sr),
        samplerate=sr,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def _is_silence(audio, threshold=AUDIO_SILENCE_THRESH):
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < threshold


def _preprocess_audio(audio, sr=AUDIO_SR):
    clip_len = int(AUDIO_CHUNK_DUR * sr)

    max_val = np.max(np.abs(audio)) + 1e-6
    audio = audio / max_val

    if len(audio) < clip_len:
        audio = np.pad(audio, (0, clip_len - len(audio)))
    else:
        audio = audio[:clip_len]

    return audio.reshape((1, clip_len, 1))


def _predict_audio_emotion(model, audio):
    """
    Returns:
        label ∈ {"Positive", "Negative"}, conf ∈ [0, 1]
    """
    features = _preprocess_audio(audio)
    prediction = model.predict(features, verbose=0)[0][0]

    if prediction > 0.5:
        label = "Negative"
        conf = float(prediction)
    else:
        label = "Positive"
        conf = float(1.0 - prediction)

    return label, conf


def start_audio_emotion_worker():
    """
    Background thread for RAVDESS audio model.
    """
    state = {
        "last_label": None,
        "last_conf": None,
        "available": True,
        "stop": False,
    }

    def worker():
        try:
            model = _load_audio_model()
        except Exception as e:
            print("[AUDIO] ERROR loading model:", e)
            state["available"] = False
            return

        print("[AUDIO] starting loop (5s chunks)")
        while not state["stop"]:
            try:
                audio = _record_audio()
            except Exception as e:
                print("[AUDIO] error recording:", e)
                state["available"] = False
                state["last_label"] = "Audio error"
                state["last_conf"] = None
                time.sleep(1.0)
                continue

            state["available"] = True

            if _is_silence(audio):
                state["last_label"] = "No speech"
                state["last_conf"] = None
                print("[AUDIO] No speech detected (skipping model)")
            else:
                try:
                    label, conf = _predict_audio_emotion(model, audio)
                    state["last_label"] = label
                    state["last_conf"] = conf
                    print(f"[AUDIO] {label} {conf*100:.1f}%")
                except Exception as e:
                    print("[AUDIO] error in prediction:", e)
                    state["last_label"] = "Audio error"
                    state["last_conf"] = None
                    state["available"] = False
                    time.sleep(1.0)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return state


def fuse_face_audio_valence(face_label, face_conf, audio_state):
    """
    Simple fusion between face (DFEW) and audio (RAVDESS).
    """
    if (
        not audio_state["available"]
        or audio_state["last_conf"] is None
        or audio_state["last_label"] not in ("Positive", "Negative")
    ):
        return face_label, face_conf

    a_label = audio_state["last_label"]
    a_conf = audio_state["last_conf"]

    if face_label not in ("Positive", "Negative"):
        return a_label, a_conf

    if face_label == a_label:
        fused_conf = min(1.0, (face_conf + a_conf) / 2.0)
        return face_label, fused_conf

    if face_conf >= a_conf + FUSION_MARGIN:
        return face_label, face_conf
    if a_conf >= face_conf + FUSION_MARGIN:
        return a_label, a_conf

    return "Uncertain", 0.0


def main():
    audio_state = start_audio_emotion_worker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    frame_buffer = deque(maxlen=ENG_BUFFER_LENGTH)
    eng_state = start_engagement_worker(frame_buffer, DEFAULT_THRESHOLDS)

    val_history = deque(maxlen=VAL_HISTORY_LENGTH)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_proc = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        frame_buffer.append(frame_proc)

        # Face valence
        face_label, face_conf = run_valence(frame_proc)

        if face_label in ("Positive", "Negative"):
            val_history.append((face_label, face_conf))

        if len(val_history) == 0:
            smooth_label = "Uncertain"
            smooth_conf = 0.0
        else:
            positives = [c for l, c in val_history if l == "Positive"]
            negatives = [c for l, c in val_history if l == "Negative"]
            p_count, n_count = len(positives), len(negatives)
            p_conf = sum(positives) / p_count if p_count else 0.0
            n_conf = sum(negatives) / n_count if n_count else 0.0

            if p_count == 0 and n_count == 0:
                smooth_label = "Uncertain"
                smooth_conf = 0.0
            else:
                if p_count > n_count or (p_count == n_count and p_conf >= n_conf):
                    smooth_label = "Positive"
                    smooth_conf = p_conf
                else:
                    smooth_label = "Negative"
                    smooth_conf = n_conf

        # Fusion
        fused_label, fused_conf = fuse_face_audio_valence(
            smooth_label, smooth_conf, audio_state
        )

        if fused_label == "Uncertain":
            val_text = "Emotion: Uncertain"
            val_conf_text = ""
        else:
            val_text = f"Emotion: {fused_label}"
            val_conf_text = f"Confidence: {fused_conf*100:.1f}%"

        # Engagement
        eng_text = f"Engagement: {eng_state['last_eng_label']}"

        # Audio status
        if not audio_state["available"]:
            audio_text = "Audio: unavailable"
        else:
            a_label = audio_state["last_label"]
            if a_label in ("Positive", "Negative"):
                audio_text = "Audio: talking"
            elif a_label == "No speech":
                audio_text = "Audio: silent"
            elif a_label is None:
                audio_text = "Audio: waiting..."
            else:
                audio_text = f"Audio: {a_label}"

        if fused_label == "Positive":
            val_color = (0, 255, 0)
        elif fused_label == "Negative":
            val_color = (0, 0, 255)
        else:
            val_color = (0, 255, 255)

        # Panel
        h, w, _ = frame_proc.shape
        panel_w = 260
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        cv2.putText(
            panel, "Model Outputs", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.putText(
            panel, val_text, (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2, cv2.LINE_AA
        )
        if val_conf_text:
            cv2.putText(
                panel, val_conf_text, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color, 1, cv2.LINE_AA
            )

        cv2.putText(
            panel, eng_text, (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.putText(
            panel, audio_text, (10, 195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA
        )

        combined = np.hstack([frame_proc, panel])
        cv2.imshow(
            "Multimodal Demo (Emotion + Engagement + Audio Activity)",
            combined,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    eng_state["stop"] = True
    audio_state["stop"] = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

