---
title: Video Anomaly Detection
emoji: 🎥
colorFrom: blue
colorTo: red
sdk: docker
sdk_version: "4.36.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Video Anomaly Detection for Avenue Dataset

This application detects anomalous events in videos using a spatio-temporal attention model.

## Usage
1. Upload a video file
2. Adjust frame skip (lower = more accurate but slower)
3. Set anomaly threshold (0.6 recommended)
4. Click "Run Detection"

## Model
The model uses a combination of:
- CNN Encoder for feature extraction
- ConvLSTM for temporal modeling  
- Spatial and Channel Attention mechanisms
- Future frame prediction with reconstruction error

## Notes
- The model expects grayscale frames resized to 128x128
- Higher scores indicate more anomalous behavior
- Processing time depends on video length and frame skip value