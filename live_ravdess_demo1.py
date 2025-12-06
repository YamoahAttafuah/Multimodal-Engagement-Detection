import numpy as np
import sounddevice as sd
import tensorflow as tf
import os

script_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_path, "RAVDESS", "audio1dconv.keras")

model=tf.keras.models.load_model(model_path)

# returns 5*22050 features shape=(110250,)
def record(dur=5, sr=22050):
    print('Recording 5 seconds of audio')
    audio = sd.rec(int(dur*sr), 
                   samplerate=sr, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    return audio.flatten()

# if True, skip model.predict
def silence(audio, threshold):
    x=np.sqrt(np.mean(audio**2))
    return x<threshold

# normalize features, ensure 5 seconds, reshape=(1,110250,1)
def preprocess(audio):
    clip_len = 5*22050
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    if len(audio) < clip_len:
        audio = np.pad(audio, (0, clip_len - len(audio)))
    else:
        audio = audio[:clip_len]
    return audio.reshape((1, clip_len, 1))

# return class prediction and confidence
def predict(audio):
    features = preprocess(audio)
    prediction = model.predict(features, verbose=0)[0][0]
    
    # prediction >0.5 is confidence in negative
    if prediction > 0.5:
        pred = 'negative'
        conf = prediction
    # prediction <0.5 is low confidence in negative
    # <=> opposite confidence in positive
    else:
        pred = 'positive'
        conf = 1 - prediction
    return pred, conf

def main():
    print("------------------")
    print('RECORDING AND INFERENCE BEGINNING SHORTLY')
    print("\nPress Ctrl+C to stop\n")
    print("------------------")
    try:
        while True:
            audio = record(dur=5, sr=22050)
            if silence(audio, 0.001):
                print('No audio detected. Skipping prediction')
                print("------------------")
            else:
                emotion, confidence = predict(audio)
                k=str(confidence*100)+'% confident'
                print(emotion,k)
                print("------------------")
                
    except KeyboardInterrupt:
        print('Stopping recording and inference')
    
    return

if __name__ == "__main__":
    main()