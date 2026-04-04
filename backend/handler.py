import os
import base64
import torch
import numpy as np
from PIL import Image
import io

from model_utils import ModelHandler

class EndpointHandler:
    def __init__(self, path=""):
        # The path parameter is provided by Hugging Face to locate the model weights
        model_path = os.path.join(path, "attention_Avenue.pth") if path else "attention_Avenue.pth"
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.handler = ModelHandler(model_path, device=device)

    def decode_image(self, b64):
        # Handle cases where the base64 string includes metadata like "data:image/jpeg;base64,"
        if ',' in b64:
            b64 = b64.split(',')[-1]
            
        image_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB') # convert to RGB to avoid mode issues, though transform converts to Grayscale
        return self.handler.transform(img)

    def __call__(self, data):
        """
        Expected input payload:
        {
            "inputs": {
                "frames": [b64_1, ..., b64_5]
            }
        }
        """
        # Hugging Face usually wraps the payload in an "inputs" key
        inputs = data.pop("inputs", data)
        frames_b64 = inputs.get("frames", [])

        if not frames_b64 or len(frames_b64) == 0:
            return {"error": "Missing 'frames' key or empty frames list"}
            
        tensors = [self.decode_image(f) for f in frames_b64]

        # Stack tensors and add batch dimension, move to model's device
        frames = torch.stack(tensors).unsqueeze(0).to(self.handler.device)  # (1, num_frames, 1, 128, 128)

        score = self.handler.predict_sequence(frames)

        return {
            "score": float(score),
            "is_anomaly": bool(score > 0.6)
        }
