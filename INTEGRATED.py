import sounddevice as sd
import numpy as np
import librosa
import cv2
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the pretrained CNN model
model = load_model(r"C:\Users\rutab\OneDrive\Desktop\BE\YEAR 2\Sem 4\SER PROJECT\emotion_recognition_model.keras")  # Path to your model

# Emotion labels (match with your trained model's output)
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust']

# Parameters
duration = 2  # seconds of audio per chunk
fs = 22050    # sampling rate

# Function to record audio
def record_audio(duration=2, fs=22050):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

# Function to get mel spectrogram
def get_mel_spectrogram(audio, sr=22050):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Predict emotion from mel spectrogram
def predict_emotion(mel_spectrogram):
    resized = cv2.resize(mel_spectrogram, (128, 128))
    norm_img = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
    input_img = norm_img.reshape(1, 128, 128, 1)
    input_img = np.repeat(input_img, 3, axis=-1)  # Convert to (1, 128, 128, 3)

    predictions = model.predict(input_img)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    if predicted_index < len(emotion_labels):
        return emotion_labels[predicted_index], confidence
    else:
        return "Unknown", confidence

# Main loop with input count
try:
    num_recognitions = int(input("🔢 Enter the number of recognitions to perform: "))
    count = 0

    print("\n🔁 Starting real-time emotion recognition...\n")
    while count < num_recognitions:
        audio = record_audio(duration, fs)
        mel_spec = get_mel_spectrogram(audio, sr=fs)
        emotion, confidence = predict_emotion(mel_spec)

        print(f"🧠 Emotion: {emotion} ({confidence*100:.2f}%)\n")

        # Optional: visualize the mel spectrogram
        plt.clf()
        plt.imshow(mel_spec, aspect='auto', origin='lower')
        plt.title(f'Emotion: {emotion} ({confidence*100:.1f}%)')
        plt.pause(0.01)

        count += 1
        time.sleep(0.5)

    print("\n✅ Completed all recognitions.")

except KeyboardInterrupt:
    print("\n🛑 Real-time prediction stopped by user.")
except Exception as e:
    print(f"\n❌ Error: {e}")
