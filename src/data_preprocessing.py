import os
import cv2
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import *
from utils import extract_dataset, main_logger as logger

def parse_filename(filename):
    parts = filename.split('_')
    if len(parts) == 3:
        try:
            subject = int(parts[0][1:])  # Remove the 's' prefix and convert to int
            emotion_code = parts[1]
            sentence = int(parts[2].split('.')[0])  # Remove the file extension
            return subject, emotion_code, sentence
        except ValueError as e:
            logger.warning(f"Error parsing filename {filename}: {str(e)}")
    return None, None, None

class EmotionDataset(Dataset):
    def __init__(self, data_dir, subjects, transform=None):
        self.data_dir = data_dir
        self.subjects = subjects
        self.transform = transform
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            logger.error(f"No samples found in {data_dir} for subjects {subjects}")

    def _load_samples(self):
        samples = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in tqdm(files, desc="Loading samples"):
                if file.endswith('.avi'):
                    subject, emotion_code, sentence = parse_filename(file)
                    if subject is not None and subject in self.subjects:
                        video_path = os.path.join(root, file)
                        if emotion_code in EMOTION_CODES:
                            emotion_index = EMOTION_CODES.index(emotion_code)
                            samples.append((video_path, emotion_index))
                        else:
                            logger.warning(f"Unknown emotion code in file: {file}")
        logger.info(f"Total samples found: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            # Load video frames
            video = self._load_video(video_path)
            
            # Extract audio
            audio = self._load_audio(video_path)
            
            # Apply transformations if any
            if self.transform:
                video = self.transform(video)
                audio = self.transform(audio)

            return video, audio, label
        except Exception as e:
            logger.error(f"Error loading data from {video_path}: {str(e)}")
            # Return a placeholder or skip this sample
            return np.zeros((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)), np.zeros(MAX_AUDIO_SAMPLES), label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(frame)
        cap.release()
        
        # Pad if necessary
        if len(frames) < MAX_FRAMES:
            padding = [frames[-1]] * (MAX_FRAMES - len(frames))
            frames.extend(padding)
        
        return np.array(frames)

    def _load_audio(self, video_path):
        try:
            audio, _ = librosa.load(video_path, sr=SAMPLE_RATE, mono=True, duration=MAX_AUDIO_DURATION)
            # Pad or truncate audio to fixed length
            if len(audio) < MAX_AUDIO_SAMPLES:
                audio = np.pad(audio, (0, MAX_AUDIO_SAMPLES - len(audio)))
            else:
                audio = audio[:MAX_AUDIO_SAMPLES]
        except Exception as e:
            logger.error(f"Error loading audio from {video_path}: {str(e)}")
            audio = np.zeros(MAX_AUDIO_SAMPLES)
        return audio

def collate_fn(batch):
    # Filter out any errored samples (where video is a scalar tensor)
    batch = [sample for sample in batch if sample[0].shape[0] > 1]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    videos, audios, labels = zip(*batch)
    videos = torch.from_numpy(np.stack(videos)).float().permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
    audios = torch.from_numpy(np.stack(audios)).float()
    labels = torch.tensor(labels)
    return videos, audios, labels

def get_data_loaders():
    # Extract dataset if not already extracted
    extract_dataset()

    # Get all subjects
    all_subjects = set()
    for root, dirs, files in os.walk(EXTRACTED_DATA_DIR):
        for file in files:
            if file.endswith('.avi'):
                subject, _, _ = parse_filename(file)
                if subject is not None:
                    all_subjects.add(subject)
    all_subjects = sorted(list(all_subjects))
    logger.info(f"Found subjects: {all_subjects}")

    if not all_subjects:
        logger.error("No subjects found in the dataset. Please check the dataset structure.")
        return None, None, None

    # Split subjects into train (80%) and test (20%) sets
    train_subjects, test_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42)
    
    # Further split train into actual train and validation sets
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = EmotionDataset(EXTRACTED_DATA_DIR, train_subjects)
    val_dataset = EmotionDataset(EXTRACTED_DATA_DIR, val_subjects)
    test_dataset = EmotionDataset(EXTRACTED_DATA_DIR, test_subjects)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error("One or more datasets are empty. Cannot create DataLoaders.")
        return None, None, None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    if train_loader is not None:
        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of validation batches: {len(val_loader)}")
        logger.info(f"Number of testing batches: {len(test_loader)}")

        # Check a sample batch
        for videos, audios, labels in train_loader:
            logger.info(f"Video batch shape: {videos.shape}")
            logger.info(f"Audio batch shape: {audios.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            break
    else:
        logger.error("Failed to create data loaders. Please check the dataset.")
