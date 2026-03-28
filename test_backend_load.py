import torch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from model_utils import ModelHandler

def test_load():
    ckpt = "scripts/attention_Avenue.pth"
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint not found at {ckpt}")
        return
    
    print(f"[INFO] Loading model from {ckpt}...")
    try:
        handler = ModelHandler(ckpt)
        print("[SUCCESS] Model loaded successfully on", handler.device)
        
        # Dummy inference
        dummy_frame = torch.randn(1, 1, 128, 128).to(handler.device)
        print("[INFO] Running dummy inference (buffer filling)...")
        for i in range(6):
            score, pred = handler.predict(dummy_frame)
            if score is not None:
                print(f"[SUCCESS] Inference result: {score:.6f}")
            else:
                print(f"[INFO] Buffering: frame {i+1}/6")
                
    except Exception as e:
        print(f"[ERROR] Failed to load or run model: {e}")

if __name__ == "__main__":
    test_load()
