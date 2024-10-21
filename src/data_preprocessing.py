import os
import cv2
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import *
from utils import extract_dataset, main_logger as logger

class EmotionDataset(Dataset):
    def __init__(self, data_dir, subjects, transform=None):
        self.data_dir = data_dir
        self.subjects = subjects
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for subject in self.subjects:
            subject_dir = os.path.join(self.data_dir, f'subject{subject}')
            for emotion in EMOTIONS:
                emotion_dir = os.path.join(subject_dir, emotion)
                for sentence in range(1, NUM_SENTENCES + 1):
                    video_path = os.path.join(emotion_dir, f'sentence{sentence}.avi')
                    samples.append((video_path, EMOTIONS.index(emotion)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Load video frames
        video = self._load_video(video_path)
        
        # Extract audio
        audio = self._load_audio(video_path)
        
        # Apply transformations if any
        if self.transform:
            video = self.transform(video)
            audio = self.transform(audio)

        return video, audio, label

    def _load_video(self, video_path):
        video = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.append(frame)
        cap.release()
        return np.array(video)

    def _load_audio(self, video_path):
        audio, _ = librosa.load(video_path, sr=16000, mono=True)
        return audio

def get_data_loaders():
    # Extract dataset if not already extracted
    extract_dataset()

    # Get all subjects
    all_subjects = list(range(1, NUM_SUBJECTS + 1))
    
    # Split subjects into train (80%) and test (20%) sets
    train_subjects, test_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42)
    
    # Further split train into actual train and validation sets
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = EmotionDataset(EXTRACTED_DATA_DIR, train_subjects)
    val_dataset = EmotionDataset(EXTRACTED_DATA_DIR, val_subjects)
    test_dataset = EmotionDataset(EXTRACTED_DATA_DIR, test_subjects)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    logger.info(f"Number of testing batches: {len(test_loader)}")

    # Check a sample batch
    for videos, audios, labels in train_loader:
        logger.info(f"Video batch shape: {videos.shape}")
        logger.info(f"Audio batch shape: {audios.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        break
