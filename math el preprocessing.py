import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
DATASET_PATH = r"C:\Users\rutab\Downloads\audios blahblah"  # Path to your dataset directory
OUTPUT_PATH = r"C:\Users\rutab\Downloads\New folder"  # Directory to save spectrogram images
LABELS_CSV_PATH = 'audio blahblah.csv'  # CSV file to save file paths and labels

# Function to create a spectrogram from an audio file
def create_spectrogram(audio_path, output_path, sample_rate=22050, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Create the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Plot and save the spectrogram as an image file
    plt.figure(figsize=(10, 4))
    plt.axis('off')  # No axes for the spectrogram image
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to process the entire dataset
def process_dataset(dataset_path=DATASET_PATH, output_path=OUTPUT_PATH):
    print(f"Processing audio dataset: {os.path.basename(os.path.normpath(dataset_path))}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    records = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Extract label from the directory name or file name (customize as needed)
                label = os.path.basename(root)  # Assuming label is the directory name
                
                # Generate the output file path for the spectrogram image
                output_file_name = os.path.splitext(file)[0] + '.png'
                output_file_path = os.path.join(output_path, output_file_name)
                
                # Create the spectrogram and save it as an image
                create_spectrogram(file_path, output_file_path)
                
                # Save the record (file path, label)
                records.append([output_file_path, label])
    
    # Convert to a DataFrame
    df = pd.DataFrame(records, columns=['file_path', 'label'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(LABELS_CSV_PATH, index=False)
    print(f"Processing complete. Spectrograms saved to {output_path}, and labels saved to {LABELS_CSV_PATH}")

def main():
    # Process the dataset and generate spectrograms
    process_dataset()

if __name__ == "__main__":
    main()
