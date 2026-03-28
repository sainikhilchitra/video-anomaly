from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from model_utils import ModelHandler
import os

app = FastAPI()

# Allow CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "../scripts/attention_Avenue.pth")
current_dir = os.path.dirname(os.path.abspath(__file__))

# Resolve path relative to script directory
if not os.path.isabs(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.abspath(os.path.join(current_dir, CHECKPOINT_PATH))

# Fallback if model not found (useful for different folder structures in Render)
if not os.path.exists(CHECKPOINT_PATH):
    local_fallback = os.path.join(current_dir, "attention_Avenue.pth")
    if os.path.exists(local_fallback):
        CHECKPOINT_PATH = local_fallback

# Initialize Model (Weights only for security)
handler = ModelHandler(CHECKPOINT_PATH)

# Simple score normalization (Moving Min-Max approximation)
class ScoreNormalizer:
    def __init__(self):
        self.min_val = 1e-5
        self.max_val = 1e-4

    def normalize(self, score):
        if score < self.min_val: self.min_val = score
        if score > self.max_val: self.max_val = score
        # Clip to [0, 1]
        norm = (score - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return min(max(norm, 0.0), 1.0)

normalizer = ScoreNormalizer()

@app.get("/")
async def root():
    return {"status": "Anomaly Detection Backend Running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("New Client Connected")
    
    try:
        while True:
            # Receive base64 frame from frontend
            data = await websocket.receive_text()
            payload = json.loads(data)
            frame_b64 = payload.get("image")
            
            if not frame_b64:
                continue
                
            # Preprocess
            frame_tensor = handler.preprocess_image(frame_b64)
            
            # Predict
            score, pred_b64 = handler.predict(frame_tensor)
            
            if score is not None:
                # Normalize for UI consumption
                norm_score = normalizer.normalize(score)
                
                # Determine status
                is_anomaly = norm_score > 0.6 # Editable threshold
                
                await websocket.send_json({
                    "score": norm_score,
                    "raw_score": score,
                    "prediction": f"data:image/png;base64,{pred_b64}",
                    "is_anomaly": is_anomaly,
                    "timestamp": payload.get("timestamp")
                })
            else:
                # Still filling buffer
                await websocket.send_json({
                    "status": "buffering",
                    "buffer_count": len(handler.history)
                })

    except WebSocketDisconnect:
        print("Client Disconnected")
        handler.history = [] # Reset buffer

if __name__ == "__main__":
    import uvicorn
    # Use $PORT provided by Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
