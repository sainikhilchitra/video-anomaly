import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64

# ============================================================
# ARCHITECTURE (From test_avenue.py)
# ============================================================

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
    def forward(self, x):
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class FutureFramePredictorWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.convlstm = ConvLSTM(128, 128)
        self.attention = SpatioTemporalAttention(128)
        self.decoder = Decoder()
    def forward(self, x):
        enc = torch.stack([self.encoder(x[:, t]) for t in range(x.size(1))], dim=1)
        h = self.convlstm(enc)
        h = self.attention(h)
        return self.decoder(h)

# ============================================================
# INFERENCE HANDLER
# ============================================================

class ModelHandler:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FutureFramePredictorWithAttention().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        
        # Buffer to store frame history
        self.history = []
        self.alpha = 0.3 # Pixel weight
        self.beta = 0.7  # Feature weight

    def preprocess_image(self, base64_str):
        image_data = base64.b64decode(base64_str.split(',')[-1])
        img = Image.open(io.BytesIO(image_data))
        return self.transform(img).to(self.device).unsqueeze(0)

    def predict(self, frame_tensor):
        """
        frame_tensor: (1, 1, 128, 128)
        returns score, prediction_vis
        """
        # Maintain history (SEQUENCE_LENGTH = 5)
        self.history.append(frame_tensor)
        if len(self.history) < 6:
            return None, None # Need at least 5 frames + 1 target
            
        # Last 5 frames for context
        sequence = torch.stack(self.history[:-1], dim=1).to(self.device) # (1, 5, 1, 128, 128)
        target = self.history[-1] # Current "actual" frame
        
        # Keep buffer to 6 (5 context + 1 comparison)
        self.history = self.history[1:]

        with torch.no_grad():
            prediction = self.model(sequence) # (1, 1, 128, 128)
            
            # 1. Pixel Level Score
            pixel_mse = torch.mean((prediction - target) ** 2).item()
            
            # 2. Feature Level Score
            feat_pred = self.model.encoder(prediction)
            feat_gt = self.model.encoder(target)
            feat_mse = torch.mean((feat_pred - feat_gt) ** 2).item()
            
            # 3. Combined Score
            combined_score = self.alpha * pixel_mse + self.beta * feat_mse
            
            # Convert prediction to base64 for visualization
            pred_img = (prediction[0, 0].cpu().numpy() * 255).astype(np.uint8)
            # Resize for better view
            pred_pil = Image.fromarray(pred_img)
            buffered = io.BytesIO()
            pred_pil.save(buffered, format="PNG")
            pred_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return combined_score, pred_base64
