# ============================================================
# train_test_avenue.py  (FULL SCRIPT)
#
# Avenue Video Anomaly Detection (Frame-Folder Pipeline)
# Model: Encoder + ConvLSTM + SpatioTemporalAttention + Decoder
# Scoring: alpha * Pixel(MSE) + beta * Feature(MSE over encoder features)
# Saves: model + logs + plots + arrays into runs/<run_name>/
#
# IMPORTANT:
# - Works on Windows with num_workers>0 because everything runs under main()
# - Expects extracted frames:
#     ../datasets/Avenue/frames/train/<video_folder>/*.jpg
#     ../datasets/Avenue/frames/test/<video_folder>/*.jpg
# - Expects Avenue GT as .mat folders:
#     ../datasets/Avenue/test_gt/1_label/*.mat
#     ../datasets/Avenue/test_gt/2_label/*.mat
#     ...
# ============================================================

import os
import re
import json
import csv
import time
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.io import loadmat


# ----------------------------
# CONFIG (edit as needed)
# ----------------------------
ROOT_DIR = "../datasets/Avenue"
FRAMES_ROOT = os.path.join(ROOT_DIR, "frames")
GT_ROOT = os.path.join(ROOT_DIR, "test_gt")

SEQUENCE_LENGTH = 5
IMAGE_SIZE = 128
BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 50  # set lower for quick debug, higher for real run
MODEL_NAME = "attention_avenue_frames"
RUNS_DIR = "runs"

ALPHA = 0.3
BETA = 0.7

# Windows-friendly workers:
NUM_WORKERS_TRAIN = 2
NUM_WORKERS_TEST = 2

# If you still face issues, set these to 0.
# NUM_WORKERS_TRAIN = 0
# NUM_WORKERS_TEST = 0


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def normalize_scores(x: List[float]) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_training_log(out_dir: str, epoch_losses: List[float]):
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "train_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "avg_loss"])
        for i, loss in enumerate(epoch_losses, 1):
            w.writerow([i, float(loss)])

    plt.figure()
    plt.plot(epoch_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Avg MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

def save_test_artifacts(out_dir: str, scores: np.ndarray, gt_labels: np.ndarray, auc_value: float,
                        fpr: np.ndarray, tpr: np.ndarray):
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "scores.npy"), scores)
    np.save(os.path.join(out_dir, "gt_labels.npy"), gt_labels)

    save_json(os.path.join(out_dir, "metrics.json"), {"auc": float(auc_value)})

    plt.figure(figsize=(12, 4))
    plt.plot(scores)
    plt.title("Normalized Anomaly Scores")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

def save_pred_vs_gt(out_dir: str, gt_img: np.ndarray, pred_img: np.ndarray, name: str):
    ensure_dir(out_dir)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(gt_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted")
    plt.imshow(pred_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_pred_vs_gt.png"), dpi=150)
    plt.close()


# ============================================================
# DATASET (FRAME FOLDERS)
# ============================================================
class AvenueFramesDataset(Dataset):
    """
    Expected folder layout:
      FRAMES_ROOT/train/<video_folder>/*.jpg
      FRAMES_ROOT/test/<video_folder>/*.jpg
    """
    def __init__(self, frames_root: str, sequence_length=5, image_size=128, mode="train"):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.mode = mode

        self.video_dir = os.path.join(frames_root, "train" if mode == "train" else "test")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.samples: List[Tuple[str, str, List[str], int]] = []
        self.video_folders: List[str] = []

        self._prepare_samples()

    def _prepare_samples(self):
        if not os.path.isdir(self.video_dir):
            raise FileNotFoundError(
                f"Frames directory not found: {self.video_dir}\n"
                f"Expected: {FRAMES_ROOT}/train and {FRAMES_ROOT}/test\n"
                f"Run frame extraction first."
            )

        self.video_folders = sorted(
            [d for d in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, d))],
            key=natural_key
        )
        if len(self.video_folders) == 0:
            raise ValueError(f"No frame folders found under: {self.video_dir}")

        for video in self.video_folders:
            video_path = os.path.join(self.video_dir, video)

            frames = sorted(
                [f for f in os.listdir(video_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))],
                key=natural_key
            )

            if len(frames) <= self.sequence_length:
                continue

            for start in range(len(frames) - self.sequence_length):
                self.samples.append((video_path, video, frames, start))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples created in {self.video_dir}. "
                f"Check that each folder has more than {self.sequence_length} frames."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, video_name, frames, start = self.samples[idx]

        input_tensor = []
        for t in range(self.sequence_length):
            img = Image.open(os.path.join(video_path, frames[start + t])).convert("L")
            img = self.transform(img)
            input_tensor.append(img)

        input_tensor = torch.stack(input_tensor, dim=0)  # (T, 1, H, W)

        target_idx = start + self.sequence_length
        target_img = Image.open(os.path.join(video_path, frames[target_idx])).convert("L")
        target_img = self.transform(target_img)

        return input_tensor, target_img, video_name, target_idx


# ============================================================
# MODEL (same architecture, includes ConvLSTM init fix)
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()

        # FIX: use hidden_dim channels for h and c
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
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
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

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
        encoded = torch.stack([self.encoder(x[:, t]) for t in range(x.size(1))], dim=1)
        h = self.convlstm(encoded)
        h = self.attention(h)
        return self.decoder(h)


# ============================================================
# GT LOADING (Avenue .mat folders) - robust inference
# ============================================================
def find_first_mat(folder_path: str):
    mats = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".mat")], key=natural_key)
    if not mats:
        return None
    return os.path.join(folder_path, mats[0])

def to_1d_int_array(x):
    x = np.array(x)
    if x.dtype == object:
        flat = x.reshape(-1)
        if len(flat) == 1:
            x = np.array(flat[0])
        else:
            raise ValueError("Object array has multiple entries; cannot convert directly to vector.")
    return x.reshape(-1).astype(np.int64)

def infer_labels_from_mat(mat_path: str, n_frames: int):
    """
    Avenue GT (.mat): key 'volLabel' is (1, T) object array.
    Each entry is a (H, W) mask (uint8). Frame is anomalous if any pixel > 0.
    Returns: labels (T,) in {0,1}.
    """
    mat = loadmat(mat_path)
    if "volLabel" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"'volLabel' not found in {mat_path}. Keys={keys}")

    v = mat["volLabel"].reshape(-1)  # (T,)
    labels = np.zeros(len(v), dtype=np.int64)

    for i in range(len(v)):
        mask = np.array(v[i])        # (H,W)
        labels[i] = 1 if np.any(mask > 0) else 0

    # Sanity align to extracted frames (should match)
    if len(labels) != n_frames:
        m = min(len(labels), n_frames)
        labels = labels[:m]

    return labels


def find_label_mat_files(gt_root: str):
    """
    Looks for .mat files inside gt_root that contain '_label' in filename.
    Example: 1_label.mat, 2_label.mat, ...
    Returns sorted list of full paths.
    """
    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    mats = [
        f for f in os.listdir(gt_root)
        if f.lower().endswith(".mat") and ("_label" in f.lower())
    ]

    if len(mats) == 0:
        raise ValueError(f"No *_label.mat files found in: {gt_root}")

    mats = sorted(mats, key=natural_key)
    return [os.path.join(gt_root, f) for f in mats]


def build_per_video_gt_from_matfiles(test_video_folders: List[str], gt_root: str, frames_test_dir: str) -> Dict[str, np.ndarray]:
    """
    Maps test videos to GT .mat files by sorted order.
    Assumes:
      - test video folders are sorted deterministically
      - GT mat files are sorted deterministically: 1_label.mat, 2_label.mat, ...
    """
    mat_paths = find_label_mat_files(gt_root)

    if len(mat_paths) < len(test_video_folders):
        raise ValueError(
            f"Not enough GT .mat files in {gt_root}: gt={len(mat_paths)} test_videos={len(test_video_folders)}"
        )

    per_video = {}
    for i, vname in enumerate(test_video_folders):
        vdir = os.path.join(frames_test_dir, vname)
        if not os.path.isdir(vdir):
            raise FileNotFoundError(f"Test frame folder missing: {vdir}")

        frames = sorted(
            [f for f in os.listdir(vdir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))],
            key=natural_key
        )
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError(f"No frames found in: {vdir}")

        mat_path = mat_paths[i]
        labels = infer_labels_from_mat(mat_path, n_frames)
        per_video[vname] = labels

    return per_video



# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Slight speed improvement on CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = f"{MODEL_NAME}_T{SEQUENCE_LENGTH}_S{IMAGE_SIZE}_bs{BATCH_SIZE}_lr{LR}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    ensure_dir(run_dir)
    print("Run dir:", run_dir)

    # ----------------------------
    # TRAIN
    # ----------------------------
    train_dataset = AvenueFramesDataset(
        frames_root=FRAMES_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE,
        mode="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS_TRAIN > 0)
    )

    model = FutureFramePredictorWithAttention().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_losses = []

    print("\n===== TRAINING =====")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, target, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

    model_path = os.path.join(run_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model -> {model_path}")

    save_training_log(run_dir, epoch_losses)
    save_json(os.path.join(run_dir, "config.json"), {
        "root_dir": ROOT_DIR,
        "frames_root": FRAMES_ROOT,
        "gt_root": GT_ROOT,
        "sequence_length": SEQUENCE_LENGTH,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": NUM_EPOCHS,
        "alpha": ALPHA,
        "beta": BETA
    })

    # ----------------------------
    # TEST + SCORES
    # ----------------------------
    print("\n===== TESTING (Pixel+Feature) =====")
    test_dataset = AvenueFramesDataset(
        frames_root=FRAMES_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE,
        mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS_TEST > 0)
    )

    attention_model = FutureFramePredictorWithAttention().to(device)
    attention_model.load_state_dict(torch.load(model_path, map_location=device))
    attention_model.eval()

    combined_scores = []
    meta = []  # (video_name, frame_idx)

    with torch.no_grad():
        for inputs, target, video_name, target_idx in tqdm(test_loader, desc="Scoring"):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = attention_model(inputs)

            pixel_err = torch.mean((pred - target) ** 2)

            feat_pred = attention_model.encoder(pred)
            feat_gt = attention_model.encoder(target)
            feature_err = torch.mean((feat_pred - feat_gt) ** 2)

            score = ALPHA * pixel_err + BETA * feature_err
            combined_scores.append(score.item())
            meta.append((video_name[0], int(target_idx.item())))

    scores = normalize_scores(combined_scores)

    # Save a score curve plot early (even before AUC)
    plt.figure(figsize=(12, 4))
    plt.plot(scores)
    plt.title("Normalized Anomaly Scores (Avenue)")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "score_curve.png"), dpi=150)
    plt.close()

    # ----------------------------
    # AUC (Avenue GT .mat)
    # ----------------------------
    print("\n===== AUC =====")
    frames_test_dir = os.path.join(FRAMES_ROOT, "test")
    test_video_folders = test_dataset.video_folders

    per_video_gt = build_per_video_gt_from_matfiles(
        test_video_folders=test_video_folders,
        gt_root=GT_ROOT,
        frames_test_dir=frames_test_dir
    )
    print("Test videos:", len(test_video_folders))
    print("GT mat files:", len(find_label_mat_files(GT_ROOT)))

    first_v = test_video_folders[0]
    print("Example mapping ->", first_v, "GT length:", len(per_video_gt[first_v]))



    gt_labels = []
    for (vname, frame_idx) in meta:
        labels = per_video_gt[vname]
        if frame_idx < 0 or frame_idx >= len(labels):
            raise ValueError(f"Frame idx out of range for {vname}: idx={frame_idx}, len={len(labels)}")
        gt_labels.append(int(labels[frame_idx]))

    gt_labels = np.array(gt_labels, dtype=np.int64)

    auc_value = roc_auc_score(gt_labels, scores)
    print(f"Avenue AUC (Attention + Pixel/Feature): {auc_value:.4f}")

    fpr, tpr, _ = roc_curve(gt_labels, scores)
    save_test_artifacts(run_dir, scores, gt_labels, auc_value, fpr, tpr)

    # ----------------------------
    # Save one qualitative sample
    # ----------------------------
    idx = min(200, len(test_dataset) - 1)
    with torch.no_grad():
        x, y, vname, fidx = test_dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        pred = attention_model(x)

        save_pred_vs_gt(
            run_dir,
            gt_img=y[0, 0].cpu().numpy(),
            pred_img=pred[0, 0].cpu().numpy(),
            name=f"sample_{vname}_frame{fidx}"
        )

    elapsed = time.time() - start_time
    print(f"\nDone. Saved outputs to: {run_dir}")
    print(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
