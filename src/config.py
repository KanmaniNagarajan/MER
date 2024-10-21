import os
import torch

# Google Drive paths
DRIVE_ROOT = '/content/drive/MyDrive/MER'
ZIPPED_DATA_PATH = os.path.join(DRIVE_ROOT, 'enterface05.zip')
EXTRACTED_DATA_DIR = os.path.join('/content', 'enterface05')
RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')

# Dataset configuration
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
NUM_SUBJECTS = 44
NUM_SENTENCES = 5

# Model configuration
AUDIO_MODEL = 'PANN_CNN14'  # or 'PANN_ResNet38'
VIDEO_MODEL = '3D_ResNet'  # or 'SlowFast'
FUSION_MODEL = 'MultimodalTransformer'

# Training configuration
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
