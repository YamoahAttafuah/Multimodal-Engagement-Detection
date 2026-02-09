# AffectFusion: Real-Time Facial Engagement Detection

https://github.com/user-attachments/assets/cf70fe0b-dd94-49dc-9d03-c4434ceb613c
Figure 1: Real-time engagement detection on live webcam feed.

AffectFusion is a multimodal framework designed to predict a person's affective state by fusing facial engagement, facial emotion, and vocal emotion signals, making it suitable for live classroom feedback, intelligent tutoring systems and AI-driven educational tools.

This repository focuses on my contribution, the facial engagement detection subsystem (DAiSEE branches), developed as part of ECE1513 Introduction to Machine Learning at the University of Toronto.

> **For complete project details, see the [full technical report](assets/ECE1513_Final_Project_Report_Team_11.pdf)**.

## 1. Problem Statement

In online education, instructors typically lack the visual cues needed to adapt lessons in real-time, as traditional feedback methods, such as surveys and polls are too slow. This work aimed to automate this feedback loop in real-time by detecting four distinct affective states, Boredom, Confusion, Engagement, and Frustration, from live video streams.

## 2. Methodology

### 2.1 Preprocessing Pipeline (```preprocessing_and_feature_extraction.py```)

The system processes the **DAISEE dataset** (9,000+ clips). To ensure robust generalization, I implemented:

1. **Subject-Independent Splitting:** A ```GroupShuffleSplit``` strategy to strictly isolate subjects between training and testing, preventing the model from memorizing faces.

2. **Feature Extraction:** Extracted 478 3D facial landmarks per frame using **MediaPipe**.

3. **Normalization and Scaling:** To make the model invariant to head position and distance to camera, all facial landmarks were normalized relative to the nose tip landmark and scaled using the interocular distance, as these are reference points that do not change significantly with different facial expressions.

4. **Feature Reduction** Applied correlation-based analysis with a 0.95 threshold to reduce input feature dimensionality from 1,434 to **54 features** and lessen computational load.

### 2.2 Model Architecture (LSTM)

A **Long Short-Term Memory (LSTM)** network was designed to analyze temporal dependencies across 75-frame video sequences of 54 facial landmark features.

* **Structure:** LSTM (64 units, 0.3 Dropout) $\rightarrow$ Dense (32 units, ReLU) $\rightarrow$ Output (Sigmoid).

* **Input:** (75 frames, 54 normalized facial landmark features)

* **Output:** Multi-label binary classification across 4 engagement states

The final model is lightweight (32k parameters, ~419 KB), and thus, is suitable for real-time inference on standard CPUs, requiring no GPU acceleration.

### 2.3 Model Training (```daisee_model_training.py```)

* **Class Imbalance:** The DAISEE dataset suffers from severe class imbalance (e.g., 'Engagement' is 10x more frequent than 'Confusion'). To resolve this, the following were implemented:

    * A Weighted Binary Cross-Entropy Loss to heavily penalize misclassification of minority classes.

    * A custom data generator to dynamically oversample minority classes for each training batch.

* **Training Callbacks:** Training callbacks applied were ModelCheckpoint, ReduceLROnPlateau and EarlyStopping, to maximize training results.

* **Threshold Adjustment:** The decision threshold for each class was independently tuned on the validation set from the default value of 0.5 to improve model performance. This
ensured the model’s sensitivity was tuned to the specific
prevalence of each emotion.

## 3. Real-Time Inference (```real-time-inference-std-top2.py```)
To demonstrate the model's immediate applicability to the real-time classroom setting:

* A live webcam demo with sub-100ms latency was built using Streamlit.
* Exponential Moving Average (EMA) was implemented for the temporal smoothing of engagement predictions.
* A visual overlay was designed to present the student's top-2 active engagement states.

## 4. Evaluation & Results

The model was evaluated on a held-out test set of unseen subjects using subject-independent splitting.

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **84.96%** |
| Model Size | 419 KB (32,676 parameters) |
| Inference Latency | < 100ms per frame |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Boredom | 0.72 | 0.68 | 0.70 |
| Confusion | 0.54 | 0.47 | 0.50 |
| Engagement | 0.91 | 0.94 | 0.92 |
| Frustration | 0.41 | 0.38 | 0.39 |

**Findings:**
- Good performance on the majority class (Engagement: 92% F1)
- Reasonable performance on Boredom despite class imbalance
- Confusion and Frustration remain challenging due to limited training samples
- Based on real-world testing of the model on a custom test set, the model generalizes reasonably well to new faces in diverse lighting conditions

## 5. Challenges & Future Work

While the system performs well in controlled environments, several challenges remain:

1. **Class Imbalance**: The model does not perform well on the minority classes, Confusion and Frustration, despite the measures taken to mitigate class imbalance. As a future step, the training could be performed on a larger more diverse dataset or on a composition of multiple datasets that provides a more balanced set of classes or labels

3. **Subtle Expressions:** The model performs best on clear expressions but occasionally misses subtle micro-expressions (e.g., slight confusion). Future work could include increasing the sampling rate or using a Transformer-based architecture could capture finer temporal nuances.

2. **Multimodal Fusion Complexity:** Currently, we use "Late Fusion" to combine audio and video signals. A future step could involve exploring "Early Fusion" techniques to allow the audio context to better inform the video predictions, though this increases computational cost.

## 6. Repository Structure

My work on the Engagement Detection subsytem can be found in the following branches:

* ```main```: Contains the source code for the DAiSEE (Engagement) model, preprocessing pipeline, and real-time inference app.

* ```daisee-lib-version```: Modularized library version of the engagement detector.

**Teammate Contributions**:
- `dfew-main`, `dfew-fold1` - Facial emotion model (Samson Ajadalu)
- `ravdess-main` - Speech emotion model (Hideki Hill)

### Key Files
```
daisee-main/
├── preprocessing_and_feature_extraction.py  # MediaPipe landmark extraction
├── daisee_model_training.py                 # LSTM training pipeline
├── real-time-inference-std-top2.py          # Streamlit live demo (dual-emotion display)
├── real-time-inference-std-top1.py          # Streamlit live demo (single-emotion display)
├── selected_features_54.json                # Feature indices (54/1434)
└── rf_daisee_model_GAMMA_54features.h5      # Trained model weights
```

## Technologies Used

**Deep Learning:** TensorFlow/Keras, LSTM networks  
**Computer Vision:** MediaPipe, OpenCV  
**Data Processing:** NumPy, Pandas, scikit-learn  
**Deployment:** Streamlit  
**Development:** Python 3.9


## Dataset

**DAiSEE (Dataset for Affective States in E-Environments)**
* **Description:** 9,000+ 10-second video clips from 112 subjects, provided by IIT Hyderabad.
* **Labels:** Each clip is annotated for 'Boredom', 'Confusion', 'Engagement', and 'Frustration'.
* **Intensity:** Labels are categorized into four levels: *Very Low, Low, High, and Very High*.
* **Source:** [IIT Hyderabad DAiSEE Project](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)

### Citation

If you use this dataset in your work, please cite the original authors:

**IEEE Style**
> [1] A. Gupta, A. D'Cunha, K. Awasthi, and V. N. Balasubramanian, "Daisee: Towards user engagement recognition in the wild," in *2016 12th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2016)*, 2016, pp. 488-495.

## Running the Demo

To run the real-time engagement detector:

```bash
# Install dependencies
pip install tensorflow opencv-python mediapipe streamlit numpy

# Run the Streamlit app
streamlit run real-time-inference-std-top2.py
```

**Note:** The demo requires the trained model file (`rf_daisee_model_GAMMA_54features.h5`) and feature indices (`selected_features_54.json`) to be in the same folder.

Besides the live webcam feed, the demo has two visual components:

**1. Video Overlay**

- A color-coded status bar above the video feed showing the current detected state.
- Displays the normalized score (0.0 – 1.0 relative to the threshold).

**2. Sidebar Metrics**

- Bars: Visual indicators of the intensity of each state.  
- Active Status: The detected states are highlighted in red with an (Active) tag.

***Attribution***

*This project was completed as part of the ECE1513 course at the University of Toronto. The full multimodal system "AffectFusion" was developed in collaboration with Samson Ajadalu (Facial Emotion) and Hideki Hill (Speech Emotion).*
