import cv2
import json
import time
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from collections import deque

# Configuration
PAGE_TITLE = "Real-Time Engagement Detection"
MODEL_PATH = 'rf_daisee_model_GAMMA_53features.h5'
JSON_PATH = 'selected_features_53.json'
SEQ_LENGTH = 75
SKIP_FRAMES = 2
EMA_ALPHA = 0.9

# Thresholds that should be calibrated
THRESHOLDS = {
    "Boredom": 0.32,
    "Engagement": 0.66,
    "Confusion": 0.21,
    "Frustration": 0.17
}

COLORS = {
    "Boredom": (0, 0, 200),      # red
    "Engagement": (0, 200, 0),   # green
    "Confusion": (0, 165, 255),  # orange
    "Frustration": (0, 0, 139)   # dark red
}

# Resource loading
@st.cache_resource
def load_resources():
    try:
        with open(JSON_PATH, 'r') as f:
            indices = np.array(json.load(f))
    except FileNotFoundError:
        st.error(f"File not found: {JSON_PATH}")
        return None, None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    return model, indices


# Function to normalize and filter landmarks
def normalize_and_filter_landmarks(landmarks, selected_indices):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    nose = coords[1]
    left_eye = coords[33]
    right_eye = coords[263]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist < 0.0001: eye_dist = 1.0
    centered = coords - nose
    normalized = centered / eye_dist
    try:
        return normalized.flatten()[selected_indices]
    except IndexError:
        return None


# Function to extract features
def extract_features(frame, face_mesh, selected_indices):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        return normalize_and_filter_landmarks(results.multi_face_landmarks[0], selected_indices)
    return None


# Function to logic & state management
def update_scores(buffer, model, current_scores):
    input_seq = np.array([buffer])
    preds = model.predict(input_seq, verbose=0)[0]
    
    raw_scores = {
        "Boredom": preds[0],
        "Confusion": preds[1],
        "Engagement": preds[2],
        "Frustration": preds[3]
    }
    
    new_scores = {}
    for label in current_scores:
        new_scores[label] = (raw_scores[label] * EMA_ALPHA) + (current_scores[label] * (1 - EMA_ALPHA))
    return new_scores


# Function to normalize scores for display in a 0-1 range where 0.5 is ALWAYS the threshold)
def normalize_score_for_display(score, threshold):
    if score < threshold:
        # Mapping [0, threshold] -> [0, 0.5]
        return 0.5 * (score / threshold)
    else:
        # Mapping [threshold, 1] -> [0.5, 1]
        return 0.5 + 0.5 * ((score - threshold) / (1.0 - threshold))


# Function to determine top active states
def get_top_states(scores):
    deviations = []
    for label, score in scores.items():
        deviation = score - THRESHOLDS.get(label, 0.5)
        if deviation > 0:
            deviations.append({"label": label, "score": score, "deviation": deviation})
    
    deviations.sort(key=lambda x: x["deviation"], reverse=True)
    
    final_states = []
    if deviations:
        top_1 = deviations[0]
        final_states.append(top_1)
        if len(deviations) > 1:
            candidate_2 = deviations[1]
            conflict_pair = {"Engagement", "Boredom"}
            current_pair = {top_1["label"], candidate_2["label"]}
            if current_pair == conflict_pair and len(deviations) > 2:
                final_states.append(deviations[2])
            else:
                final_states.append(candidate_2)
    return final_states


# Function to draw overlays
def draw_overlay(frame, active_states):
    if active_states:
        # Only show top-1 in overlay
        top_state = active_states[0]
        label = top_state['label']
        raw_val = top_state['score']
        thresh = THRESHOLDS[label]
        
        # Calculate normalized value for overlay display
        norm_val = normalize_score_for_display(raw_val, thresh)
        status_msg = f"{label.upper()} ({norm_val:.2f})"

        color = COLORS.get(label, (200, 200, 200))
        
        # Draw background rectangle
        cv2.rectangle(frame, (0, 0), (640, 60), color, -1)
        
        # Draw text
        cv2.putText(frame, status_msg, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Neutral state
        cv2.rectangle(frame, (0, 0), (640, 60), (200, 200, 200), -1)
        cv2.putText(frame, "NEUTRAL", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return frame


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    model, selected_indices = load_resources()
    if model is None:
        st.stop()

    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
        
    with col2:
        # Setup placeholders
        metrics_placeholders = {}
        for label in THRESHOLDS.keys():
            lbl = st.empty()
            bar = st.progress(0)
            metrics_placeholders[label] = {"text": lbl, "bar": bar}

    run_app = st.toggle('Start Camera', value=True)
    
    if run_app:
        cap = cv2.VideoCapture(0)
        input_buffer = deque(maxlen=SEQ_LENGTH)
        curr_scores = {k: 0.0 for k in THRESHOLDS.keys()}
        frame_count = 0
        
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while cap.isOpened() and run_app:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break
                
                frame_count += 1
                features = extract_features(frame, face_mesh, selected_indices)
                
                if features is not None:
                    input_buffer.append(features)
                elif len(input_buffer) > 0:
                    input_buffer.append(input_buffer[-1])
                else:
                    input_buffer.append(np.zeros(len(selected_indices)))

                if len(input_buffer) == SEQ_LENGTH and frame_count % SKIP_FRAMES == 0:
                    curr_scores = update_scores(input_buffer, model, curr_scores)

                active_states = get_top_states(curr_scores)
                
                # Determine Top-1 Active Label
                if active_states:
                    top_active_label = active_states[0]['label']
                else:
                    top_active_label = None
                
                # Update sidebar
                for label, score in curr_scores.items():
                    # Update bar
                    visual_val = normalize_score_for_display(score, THRESHOLDS[label])
                    visual_val = max(0.0, min(visual_val, 1.0))
                    metrics_placeholders[label]["bar"].progress(visual_val)
                    
                    # Update label text
                    is_top_active = (label == top_active_label)
                    
                    if is_top_active:
                        txt = f"**{label}** :red[(Active)]"
                    else:
                        txt = f"**{label}**"
                        
                    metrics_placeholders[label]["text"].markdown(txt)

                final_frame = draw_overlay(frame, active_states)
                frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

if __name__ == "__main__":
    main()