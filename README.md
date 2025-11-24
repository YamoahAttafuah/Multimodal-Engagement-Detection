# Real-Time Engagement Detection (DAiSEE)

This **daisee-main** branch contains the scripts for engagement detection. It utilizes an LSTM trained on the **DAiSEE dataset**, processing video input via **MediaPipe Face Mesh** to detect four affective states: **Boredom, Engagement, Confusion, and Frustration**.

---

## 1. Repository Contents

- **`real-time-inference-std-top1.py`**  
  A real-time inference script that displays only the dominant emotion (*Top-1*) in the video overlay and sidebar.

- **`real-time-inference-std-top2.py`**  
  A second real-time inference script that displays the top two active emotions (when applicable) for finer analysis.

- **`rf_daisee_model_GAMMA_53features.h5`**  
  The trained model’s weights.

- **`selected_features_53.json`**  
  The specific MediaPipe face mesh indices required by the model.

---

## 2. Prerequisites

1. Install **Python 3.11** alongside your current version of Python.
2. Create a virtual environment using **Python 3.11** and activate it.
3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. How to Run

Navigate to the project directory and run the desired Streamlit app.

### **Single-emotion display**

```bash
streamlit run real-time-inference-std-top1.py
```

### **Dual-emotion display**

```bash
streamlit run real-time-inference-std-top2.py
```

---

## 4. Features & Visualization

The application provides a real-time video feed with two visual components:

### **1. Video Overlay**

- A color-coded status bar above the video feed showing the current detected state.
- Displays the **Normalized Score** (0.0 – 1.0 relative to the threshold).

### **2. Sidebar Metrics**

- **Bars:** Visual indicators of the intensity of each state.  
- **Active Status:** The detected state(s) are highlighted in **red** with an **(Active)** tag.

---

## 5. Configuration

You can and should adjust sensitivity thresholds in the Python scripts under the `THRESHOLDS` dictionary:

```python
THRESHOLDS = {
    "Boredom": 0.35,      # Lower value = more sensitive
    "Engagement": 0.62,
    "Confusion": 0.22,
    "Frustration": 0.20
}
```

---

## 6. Common Issues

- **Model Not Found**  
  Ensure `rf_daisee_model_GAMMA_53features.h5` and `selected_features_53.json` are in the **same directory** as the script.

- **Camera Access**  
  If the browser asks for permission to access the camera, click **Allow**.  
  If the video does not start, ensure no other application (Zoom, Teams, etc.) is already using the webcam.

---
