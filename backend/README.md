# Video Anomaly Detection Model

## Input
JSON:
{
  "inputs": {
    "frames": [base64_frame1, ..., base64_frame5]
  }
}

## Output
{
  "score": float,
  "is_anomaly": boolean
}