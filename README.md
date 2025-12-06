# RAVDESS live audio emotion predictor

Binary emotion classifier trained on python 3.9.23, tensorflow 2.15.0. Source preprocessing and training notebook for submission found in RAVDESS.

## Py and Tf versions
Create conda environment:
```
conda create -n tf_env python=3.9.23
conda activate tf_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script from the conda prompt:
```
python live_ravdess_demo1.py
```
Make sure you are running with **python** not the system **py**. py uses the latest python and will not cooperate with tf 2.15.0
## Model

The trained model is located in `RAVDESS/audio1dconv.keras`
