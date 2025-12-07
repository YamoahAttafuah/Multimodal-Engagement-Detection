# DFEW Facial Emotion Model

This folder contains the **DFEW facial emotion baseline** used in the main AffectFusion project.

- 7-way emotion classification on DFEW (**Happy, Sad, Neutral, Angry, Surprise, Disgust, Fear**)
- Backbone: **ResNet-101**
- Input: pre-extracted frames (`clip_224x224_16f`)
- Clip-level prediction: average logits over frames  
- Metrics: **WAR** (accuracy) and **UAR** (macro recall)  
- Best checkpoint selected by **validation UAR**

---

## Files

- `train_dfew_16f.py` – training script (train/val on DFEW, 16-frame clips)
- `test_dfew_16f.py` – test script (evaluates on DFEW test split)
- `inference_dfew_2cls.py` – loads a pretrained 2-class (Positive/Negative) model for the main demo
- `pretrained_weight/` – saved checkpoints (see link below)

Pretrained checkpoint (7-class model) will be shared via Google Drive:

> **Download:** _[[Google Drive link here](https://drive.google.com/file/d/14BUmlN3PLFsbuyaPsYW4kuaVoBaW7I1o/view?usp=sharing)]_  
> Place the `.pth` file inside `pretrained_weight/`.

---

## Requirements

From the project root:

```bash
pip install -r dfew_emotion_code/requirements.txt
```

(PyTorch / torchvision versions should match your local setup.)

---

## Dataset (DFEW)

Follow the official instructions to obtain **DFEW Part 2** (frames + splits).

Expected layout relative to the project root (`AffectFusion_Project/`):

```text
AffectFusion_Project/
  ├─ dfew_emotion_code/
  │    ├─ train_dfew_16f.py
  │    ├─ test_dfew_16f.py
  │    ├─ inference_dfew_2cls.py
  │    └─ pretrained_weight/
  └─ dfew/DFEW-part2/
       ├─ Clip/clip_224x224_16f/<clip_id>/*.jpg
       └─ EmoLabel_DataSplit/
            ├─ train(single-labeled)/set_1.csv ... set_5.csv
            └─ test(single-labeled)/set_1.csv ... set_5.csv
```

By default, both scripts expect the dataset at:

```text
../dfew/DFEW-part2/
```

(relative to `dfew_emotion_code/`).

---

## Train

From `AffectFusion_Project/dfew_emotion_code/`:

```bash
python train_dfew_16f.py --fold 1 --epochs 30
```

This saves the best model (by **VAL UAR**) to:

```text
pretrained_weight/best_resnet101_2cls_fold1_16f_VAL.pth
```

You can change the output folder with `--out`, and the dataset root with `--root`.

---

## Test

```bash
python test_dfew_16f.py --fold 1
```

By default this loads:

```text
pretrained_weight/best_resnet101_2cls_fold1_16f_VAL.pth
```

and prints final performance:

```text
[TEST] WAR 0.xxxx  UAR 0.xxxx
```
