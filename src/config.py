import os
import torch

# Google Drive paths
DRIVE_ROOT = '/content/drive/MyDrive/MER'
ZIPPED_DATA_PATH = os.path.join(DRIVE_ROOT, 'enterface_database.zip')
EXTRACTED_DATA_DIR = '/content/enterface_database'
RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')

# Dataset configuration
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
EMOTION_CODES = ['an', 'di', 'fe', 'ha', 'sa', 'su']
NUM_SUBJECTS = 44
NUM_SENTENCES = 5

# Video configuration
MAX_FRAMES = 100
FRAME_HEIGHT = 144
FRAME_WIDTH = 180

# Audio configuration
SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 5  # in seconds
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION

# Model configuration
AUDIO_MODEL = 'PANN_CNN14'
VIDEO_MODEL = '3D_ResNet'
FUSION_MODEL = 'MultimodalTransformer'

# Training configuration
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
