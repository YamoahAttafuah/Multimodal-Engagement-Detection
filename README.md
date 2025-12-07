# AffectFusion (Demo)

Webcam-based **multimodal emotion demo** combining:

- **Facial valence** (DFEW, 2-class: Positive / Negative)  
- **Engagement level** (DAiSEE: Boredom / Engagement / Confusion / Frustration / Neutral)  
- **Speech valence** (RAVDESS audio, 2-class, fused with face into one final “Emotion”)

The main entry point is:

```bash
python main_demo.py
```

This opens your webcam and microphone and shows a live overlay with:

- **Emotion:** fused face + audio valence (Positive / Negative / Uncertain)  
- **Confidence:** scalar confidence for the fused label  
- **Engagement:** DAiSEE engagement label  
- **Audio:** talking / silent / waiting / unavailable  

---

## 1. Folder layout (demo level)

Expected layout around this folder:

```text
AffectFusion_Project/
├─ main_demo.py
├─ crop_utils.py
├─ requirements.txt
├─ daisee_engagement_code/
│    ├─ engagement_detector_lib.py
│    ├─ selected_features_53.json
│    ├─ pretrained_weight/
│    │    └─ rf_daisee_model_GAMMA_53features.h5
│    └─ ... (training code, README, etc.)
├─ dfew_emotion_code/
│    ├─ inference_dfew_2cls.py
│    ├─ train_dfew_16f.py        
│    ├─ test_dfew_16f.py         
│    ├─ pretrained_weight/
│    │    ├─ best_resnet101_2cls_fold1_16f_VAL.pth   # 2-class valence (used by demo)
│    │    
│    └─ ...
├─ ravdess_audio_code/
│    ├─ pretrained_weight/
│    │    └─ audio1dconv.keras   # 1D Conv speech model (2-class)
│    └─ ...
└─ ...
```


---

## 2. Components

### 2.1 `main_demo.py`

- Opens webcam (`cv2.VideoCapture(0)`) at 640×360.
- Imports:
  - `dfew_emotion_code/inference_dfew_2cls.py` for facial valence
  - `daisee_engagement_code/engagement_detector_lib.py` for engagement
  - `ravdess_audio_code/pretrained_weight/*.keras` for audio valence
- Runs three background workers:
  - **DAiSEE engagement** (operates on a rolling buffer of 75 frames)
  - **RAVDESS audio emotion** (5-second chunks, skips silence)
- Main loop: face crop → DFEW valence → smoothing → fusion with audio
- Draws a side panel:
  - Emotion: `{Positive / Negative / Uncertain}`
  - Confidence: `xx.x%` (fused face + audio)
  - Engagement: `{DAiSEE label}`
  - Audio: `{talking / silent / waiting ...}`

Quit with `q`.

### 2.2 `crop_utils.py`

- Uses OpenCV to detect the largest face in the frame.
- Returns a padded BGR face crop suitable for the DFEW model.

### 2.3 `dfew_emotion_code/inference_dfew_2cls.py`

- Wraps a **ResNet-101** trained on DFEW valence (2 classes).  
- On first call:
  - selects device (`cuda` / `mps` / `cpu`),
  - loads:

```text
dfew_emotion_code/pretrained_weight/best_resnet101_2cls_fold1_16f_VAL.pth
```

  - sets the model to eval mode.

API used by the demo:

```python
label, conf, probs = predict_dfew_valence(face_bgr)
# label ∈ {"Positive", "Negative"}
# conf  ∈ [0, 1] (probability of label)
# probs = [p_pos, p_neg]
```

### 2.4 `daisee_engagement_code/engagement_detector_lib.py`

- Uses **MediaPipe FaceMesh** to extract 3D landmarks.
- Reduces them to 53 geometric features and feeds a trained model:

```text
daisee_engagement_code/pretrained_weight/rf_daisee_model_GAMMA_53features.h5
daisee_engagement_code/selected_features_53.json
```

The main function:

```python
label = get_engagement_label(frames, thresholds=None)
# label ∈ {"Engagement", "Boredom", "Confusion", "Frustration", "Neutral"}
```

where `frames` is a list (or numpy array) of BGR frames.

### 2.5 `ravdess_audio_code/pretrained_weight/*.keras`

- 1D Conv speech emotion model trained on **RAVDESS**.  
- `main_demo.py` automatically picks the first `.keras` file in:

```text
ravdess_audio_code/pretrained_weight/
```

The helper `_predict_audio_emotion(...)` returns:

```python
label ∈ {"Positive", "Negative"}
conf  ∈ [0, 1]
```

- Audio is recorded as 5-second mono segments at 22.05 kHz (`sounddevice`), normalized, then passed to the model.

### 2.6 Fusion rule (face + audio)

In `main_demo.py`:

- If audio is unavailable / no speech → use **face only**.
- If face is *Uncertain* but audio is Positive/Negative → **trust audio**.
- If both are valid and agree → same label, **mean** of the two confidences.
- If they disagree:
  - pick the one with significantly higher confidence (margin = 0.10), or
  - output **Uncertain** if they’re too close.

---

## 3. Pretrained weights

For the demo to run end-to-end:

### 3.1 DFEW valence (2-class)

Download the ResNet-101 valence checkpoint:

```text
best_resnet101_2cls_fold1_16f_VAL.pth
```

Place it at:

```text
dfew_emotion_code/pretrained_weight/best_resnet101_2cls_fold1_16f_VAL.pth
```

[Google Drive link ](https://drive.google.com/file/d/14BUmlN3PLFsbuyaPsYW4kuaVoBaW7I1o/view?usp=sharing)

### 3.2 DAiSEE engagement


```text
daisee_engagement_code/pretrained_weight/rf_daisee_model_GAMMA_53features.h5
daisee_engagement_code/selected_features_53.json
```

### 3.3 RAVDESS audio

Place your `.keras` audio model in:

```text
ravdess_audio_code/pretrained_weight/
```

The demo will automatically load the first `*.keras` file it finds there.

---

## 4. Environment / setup

From `AffectFusion_Project/`:

```bash
pip install -r requirements.txt
```

Requirements include (at minimum):

- `opencv-python`
- `numpy`
- `sounddevice`
- `tensorflow`
- `torch`, `torchvision`
- `mediapipe` (for DAiSEE engagement)
- `pillow`, `scikit-learn`

You can also use a **virtualenv** / **conda** env if you prefer.

---

## 5. Running the demo

From `AffectFusion_Project/`:

```bash
python main_demo.py
```
