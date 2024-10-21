import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel
from config import *

class AudioModel(nn.Module):
    def __init__(self, model_name=AUDIO_MODEL):
        super(AudioModel, self).__init__()
        if model_name == 'PANN_CNN14':
            self.model = AutoModel.from_pretrained("microsoft/wavlm-base")
        elif model_name == 'PANN_ResNet38':
            self.model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
        else:
            raise ValueError(f"Unsupported audio model: {model_name}")
        
        self.fc = nn.Linear(self.model.config.hidden_size, 128)

    def forward(self, x):
        outputs = self.model(x).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        return self.fc(pooled)

class VideoModel(nn.Module):
    def __init__(self, model_name=VIDEO_MODEL):
        super(VideoModel, self).__init__()
        if model_name == '3D_ResNet':
            self.model = models.video.r3d_18(pretrained=True)
        elif model_name == 'SlowFast':
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        else:
            raise ValueError(f"Unsupported video model: {model_name}")
        
        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 128)

    def forward(self, x):
        return self.model(x)

class MultimodalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super(MultimodalTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, len(EMOTIONS))

    def forward(self, audio_features, video_features):
        # Combine audio and video features
        x = torch.cat((audio_features.unsqueeze(0), video_features.unsqueeze(0)), dim=0)
        
        # Pass through transformer
        x = self.transformer(x, x)
        
        # Average pooling across sequence dimension
        x = torch.mean(x, dim=0)
        
        # Final classification
        return self.fc(x)

class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.audio_model = AudioModel()
        self.video_model = VideoModel()
        self.fusion_model = MultimodalTransformer()

    def forward(self, audio, video):
        audio_features = self.audio_model(audio)
        video_features = self.video_model(video)
        return self.fusion_model(audio_features, video_features)

# Example usage
if __name__ == "__main__":
    model = EmotionRecognitionModel()
    print(model)
    
    # Test with random inputs
    audio_input = torch.randn(1, 1, 16000)  # Assuming 1 second of audio at 16kHz
    video_input = torch.randn(1, 3, 16, 112, 112)  # Assuming 16 frames of 112x112 RGB images
    output = model(audio_input, video_input)
    print(f"Output shape: {output.shape}")
