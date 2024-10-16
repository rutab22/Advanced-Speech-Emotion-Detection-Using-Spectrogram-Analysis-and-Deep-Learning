import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Constants
IMAGE_SIZE = (128, 128)  # Update based on your spectrogram image size
MODEL_PATH = 'emotion_recognition_model.keras'  # Path to the trained model
SPECTROGRAM_DIR = r"C:\Users\rutab\Downloads\New folder"  # Path to the directory containing spectrograms

# Load the trained model
model = load_model(MODEL_PATH)

# Function to preprocess an image for the model
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE, color_mode='rgb')
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict emotion from a single image
def predict_emotion(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    return predictions

# Function to process a directory of spectrograms
def process_spectrogram_directory(directory_path):
    results = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".png"):  # Assuming spectrograms are in PNG format
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                # Predict emotion
                predictions = predict_emotion(file_path)
                predicted_class = np.argmax(predictions, axis=1)
                
                # Map class index to label (assumes labels are in order)
                emotion_labels = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'}
                predicted_emotion = emotion_labels.get(predicted_class[0], 'Unknown')
                
                # Append results
                results.append({
                    'file_path': file_path,
                    'predicted_emotion': predicted_emotion,
                    'confidence': np.max(predictions)
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('prediction_results.csv', index=False)
    print("Processing complete. Results saved to 'prediction_results.csv'.")

def main():
    process_spectrogram_directory(SPECTROGRAM_DIR)

if __name__ == "__main__":
    main()
