import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import base64
from PIL import Image
import io

from model_utils import ModelHandler

app = FastAPI(title="Video Anomaly Detection WS API")

# Allow all CORS since it's a WS server connecting to a separate React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
handler = ModelHandler("attention_Avenue.pth")

def decode_image(b64_string):
    """Decodes a base64 string into a tensor ready for the ModelHandler."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[-1]
    
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    return handler.transform(img)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # We need sequences of 5 frames for the model to predict the context
    SEQUENCE_LENGTH = 5
    buffer = []
    
    try:
        while True:
            # Receive data from frontend
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            base64_image = data.get("image")
            if not base64_image:
                continue
            
            # Decode and append to sliding window buffer
            tensor = decode_image(base64_image)
            buffer.append(tensor)
            
            # Keep only the latest 5 frames
            if len(buffer) > SEQUENCE_LENGTH:
                buffer.pop(0)
                
            # If we have enough frames, predict
            if len(buffer) == SEQUENCE_LENGTH:
                # Shape required: (Batch=1, Seq=5, C, H, W)
                frames_tensor = torch.stack(buffer).unsqueeze(0).to(handler.device)
                
                score = handler.predict_sequence(frames_tensor)
                
                # The React frontend threshold defaults to 0.8
                is_anomaly = score > 0.8
                
                await websocket.send_json({
                    "score": float(score),
                    "is_anomaly": bool(is_anomaly),
                    "timestamp": data.get("timestamp")
                })
            else:
                # Still building buffer, send a neutral 0 score 
                await websocket.send_json({
                    "score": 0.0,
                    "is_anomaly": False,
                    "msg": "Buffering initial frames...",
                    "buffer_size": len(buffer)
                })
                
    except WebSocketDisconnect:
        print("Frontend disconnected.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

@app.get("/")
def read_root():
    return {"status": "Active", "type": "WebSocket Server"}

if __name__ == "__main__":
    import uvicorn
    # When deployed to Hugging Face Docker, we bind to 0.0.0.0:7860
    uvicorn.run(app, host="0.0.0.0", port=7860)