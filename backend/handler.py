import base64
import torch
from PIL import Image
import io

from model_utils import ModelHandler

handler = ModelHandler("attention_Avenue.pth")


def decode_image(b64):
    if ',' in b64:
        b64 = b64.split(',')[-1]

    image_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    return handler.transform(img)


def predict(inputs):
    if "inputs" in inputs:
        inputs = inputs["inputs"]

    frames_b64 = inputs.get("frames", [])
    threshold = inputs.get("threshold", 0.6)

    if len(frames_b64) == 0:
        return {"error": "No frames provided"}

    tensors = [decode_image(f) for f in frames_b64]
    frames = torch.stack(tensors).unsqueeze(0).to(handler.device)

    score = handler.predict_sequence(frames)

    return {
        "score": float(score),
        "threshold": threshold,
        "is_anomaly": bool(score > threshold)
    }