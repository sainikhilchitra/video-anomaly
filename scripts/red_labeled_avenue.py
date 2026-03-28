# ============================================================
# export_redbox_avenue_attention.py
#
# Avenue - Export red-box anomaly localization videos (Attention model)
# Uses extracted frame folders (fast, no AVI reading).
#
# Expected frames:
#   ../datasets/Avenue/frames/test/<video_folder>/*.jpg|png|tif
#
# Output:
#   boxed_videos_avenue_attention/<video_folder>.mp4
#
# Boxes are derived from absolute pixel error heatmap (pred vs GT frame).
# Uses connected components (robust) and keeps top-1 box per frame.
#
# IMPORTANT:
# - This is localization-by-error (unsupervised). It is NOT using GT masks to localize.
# - It will draw boxes for every frame (some may be empty if thresholding yields none).
# ============================================================

import os
import re
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = "../datasets/Avenue"
FRAMES_TEST_DIR = os.path.join(ROOT_DIR, "frames", "test")

SEQUENCE_LENGTH = 5
IMAGE_SIZE = 128

CKPT_PATH = "attention_Avenue.pth"   # update if different
OUTPUT_ROOT = "boxed_videos_avenue_attention"

FPS = 10
THR = 0.35
MIN_AREA = 20
KEEP_TOP_K = 1

# score weights (only for printing/debug)
ALPHA = 0.3
BETA = 0.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2  # Windows-safe if dataset is in this file and main() guard is used


# ============================================================
# UTILS
# ============================================================
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def normalize_01(x: np.ndarray):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def heatmap_to_boxes_connected_components(
    heat_01,         # HxW float in [0,1]
    thr=0.35,
    min_area=20,
    keep_top_k=1
):
    heat_01 = heat_01.astype(np.float32)
    mask = (heat_01 >= thr).astype(np.uint8)  # 0/1

    selected = int(mask.sum())
    total = mask.size
    selected_ratio = selected / (total + 1e-8)

    if selected == 0:
        dbg = {
            "thr": float(thr),
            "heat_min": float(heat_01.min()),
            "heat_max": float(heat_01.max()),
            "heat_mean": float(heat_01.mean()),
            "selected_px": selected,
            "selected_ratio": float(selected_ratio),
            "components": 0,
            "boxes": 0
        }
        return [], (mask * 255).astype(np.uint8), dbg

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h), int(area)))

    boxes.sort(key=lambda b: b[4], reverse=True)
    boxes = boxes[:keep_top_k]
    boxes_out = [(x, y, w, h) for (x, y, w, h, area) in boxes]

    dbg = {
        "thr": float(thr),
        "heat_min": float(heat_01.min()),
        "heat_max": float(heat_01.max()),
        "heat_mean": float(heat_01.mean()),
        "selected_px": selected,
        "selected_ratio": float(selected_ratio),
        "components": int(num_labels - 1),
        "boxes": len(boxes_out)
    }
    return boxes_out, (mask * 255).astype(np.uint8), dbg

def open_writer_for_video(output_root, video_name, frame_w, frame_h, fps):
    out_path = os.path.join(output_root, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h)), out_path


# ============================================================
# DATASET (Avenue frames)
# Returns: inputs (T,1,H,W), target (1,H,W), video_name, target_frame_name
# ============================================================
class AvenueFramesDataset(Dataset):
    def __init__(self, frames_test_dir, sequence_length=5, image_size=128):
        self.frames_test_dir = frames_test_dir
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        video_folders = sorted(
            [d for d in os.listdir(self.frames_test_dir) if os.path.isdir(os.path.join(self.frames_test_dir, d))],
            key=natural_key
        )

        for video in video_folders:
            vpath = os.path.join(self.frames_test_dir, video)

            frames = sorted(
                [f for f in os.listdir(vpath) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))],
                key=natural_key
            )
            if len(frames) <= self.sequence_length:
                continue

            for start in range(len(frames) - self.sequence_length):
                input_frames = frames[start:start + self.sequence_length]
                target_frame = frames[start + self.sequence_length]
                self.samples.append((vpath, video, input_frames, target_frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vpath, video_name, input_frames, target_frame = self.samples[idx]

        xs = []
        for f in input_frames:
            img = Image.open(os.path.join(vpath, f)).convert("L")
            xs.append(self.transform(img))
        x = torch.stack(xs, dim=0)  # (T,1,H,W)

        y = Image.open(os.path.join(vpath, target_frame)).convert("L")
        y = self.transform(y)  # (1,H,W)

        return x, y, video_name, target_frame


# ============================================================
# MODEL (Attention) - same as yours
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
# MAIN
# ============================================================
def main():
    print("Device:", DEVICE)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    dataset = AvenueFramesDataset(
        frames_test_dir=FRAMES_TEST_DIR,
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    model = FutureFramePredictorWithAttention().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    writer = None
    current_video_name = None

    with torch.no_grad():
        for idx, (inputs, target, video_name, target_frame_name) in enumerate(
            tqdm(loader, desc="Export red-box videos (Avenue Attention)")
        ):
            inputs = inputs.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            # new folder => new mp4
            if current_video_name != video_name[0]:
                if writer is not None:
                    writer.release()
                    writer = None

                current_video_name = video_name[0]
                writer, out_path = open_writer_for_video(OUTPUT_ROOT, current_video_name, IMAGE_SIZE, IMAGE_SIZE, FPS)
                print(f"\nWriting: {out_path}")

            pred = model(inputs)

            # Optional debug score (for logs only)
            pixel_err = torch.mean((pred - target) ** 2)
            feat_pred = model.encoder(pred)
            feat_gt = model.encoder(target)
            feature_err = torch.mean((feat_pred - feat_gt) ** 2)
            score = ALPHA * pixel_err + BETA * feature_err

            # heatmap from absolute pixel error
            err_map = (pred - target).abs()[0, 0].detach().cpu().numpy()
            heat_01 = normalize_01(err_map)

            boxes, mask_u8, dbg = heatmap_to_boxes_connected_components(
                heat_01,
                thr=THR,
                min_area=MIN_AREA,
                keep_top_k=KEEP_TOP_K
            )

            if idx % 500 == 0:
                print(
                    f"[{current_video_name}] idx={idx} dbg={dbg} "
                    f"err(min={err_map.min():.6f}, max={err_map.max():.6f}, mean={err_map.mean():.6f}) "
                    f"score={score.item():.6f}"
                )

            # draw on GT frame
            gt_u8 = (target[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
            final = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)

            for (x, y, w, h) in boxes:
                cv2.rectangle(final, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(final, f"{current_video_name} {target_frame_name[0]}", (6, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(final, f"boxes={len(boxes)} thr={THR}", (6, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(final)

    if writer is not None:
        writer.release()

    print(f"\nDone. Videos saved in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
