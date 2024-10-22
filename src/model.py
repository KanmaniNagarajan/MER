import torch
import torch.nn as nn
from config import *

class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.audio_model = nn.Sequential(
            nn.Linear(MAX_AUDIO_SAMPLES, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.video_model = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(16, 128),
            nn.ReLU()
        )
        self.fusion_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(EMOTIONS))
        )

    def forward(self, audio, video):
        # Audio: (batch_size, MAX_AUDIO_SAMPLES)
        audio_features = self.audio_model(audio)
        
        # Video: (batch_size, 3, MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
        video_features = self.video_model(video)
        
        combined_features = torch.cat((audio_features, video_features), dim=1)
        output = self.fusion_model(combined_features)
        return output
