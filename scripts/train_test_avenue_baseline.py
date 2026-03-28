# ============================================================
# train_test_avenue_baseline_pixelonly.py  (FULL SCRIPT)
#
# Avenue Video Anomaly Detection (Frame-Folder Pipeline)
# BASELINE MODEL (NO ATTENTION)
# Scoring: Pixel MSE ONLY (pred vs GT frame)
# Metric: ROC-AUC using Avenue GT (.mat volLabel masks)
#
# Saves into: runs/<run_name>/
#   - model.pth
#   - config.json
#   - train_log.csv
#   - loss_curve.png
#   - scores.npy
#   - gt_labels.npy
#   - metrics.json
#   - score_curve.png
#   - roc_curve.png
#   - sample_<video>_frame<idx>_pred_vs_gt.png
#
# Expects extracted frames:
#   ../datasets/Avenue/frames/train/<video_folder>/*.jpg
#   ../datasets/Avenue/frames/test/<video_folder>/*.jpg
#
# Expects Avenue GT as .mat files inside:
#   ../datasets/Avenue/test_gt/1_label.mat ... 21_label.mat
# where each .mat has 'volLabel' = (1,T) object array of masks (H,W)
#
# Windows: safe with num_workers>0 since everything is under main().
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
NUM_EPOCHS = 50  # increase for real training

MODEL_NAME = "baseline_avenue_pixelonly"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

# Windows-friendly workers (increase if stable)
NUM_WORKERS_TRAIN = 2
NUM_WORKERS_TEST = 2
# If you still face worker issues, set to 0:
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
    plt.title("Training Loss (Baseline Pixel-only)")
    plt.xlabel("Epoch")
    plt.ylabel("Avg MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

def save_test_artifacts(out_dir: str, scores: np.ndarray, gt_labels: np.ndarray,
                        auc_value: float, fpr: np.ndarray, tpr: np.ndarray):
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "scores.npy"), scores)
    np.save(os.path.join(out_dir, "gt_labels.npy"), gt_labels)
    save_json(os.path.join(out_dir, "metrics.json"), {"auc": float(auc_value)})

    plt.figure(figsize=(12, 4))
    plt.plot(scores)
    plt.title("Normalized Anomaly Scores (Baseline Pixel-only)")
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
    plt.title("ROC Curve (Baseline Pixel-only)")
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
    Expected:
      FRAMES_ROOT/train/<video_folder>/*.jpg
      FRAMES_ROOT/test/<video_folder>/*.jpg
    Returns:
      input_tensor: (T,1,H,W)
      target_img:  (1,H,W)
      video_name:  folder name (e.g., "01")
      target_idx:  int index in that video folder
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
                f"Expected: {FRAMES_ROOT}/train and {FRAMES_ROOT}/test"
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

        input_tensor = torch.stack(input_tensor, dim=0)

        target_idx = start + self.sequence_length
        target_img = Image.open(os.path.join(video_path, frames[target_idx])).convert("L")
        target_img = self.transform(target_img)

        return input_tensor, target_img, video_name, target_idx


# ============================================================
# BASELINE MODEL (NO ATTENTION)
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
        gates = self.conv(torch.cat([x, h], dim=1))
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
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class FutureFramePredictorBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.convlstm = ConvLSTM(128, 128)
        self.decoder = Decoder()

    def forward(self, x):
        encoded = torch.stack([self.encoder(x[:, t]) for t in range(x.size(1))], dim=1)
        h = self.convlstm(encoded)
        return self.decoder(h)


# ============================================================
# GT LOADING (Avenue .mat files with volLabel masks)
# ============================================================
def find_label_mat_files(gt_root: str):
    mats = sorted(
        [f for f in os.listdir(gt_root) if f.lower().endswith(".mat") and "_label" in f.lower()],
        key=natural_key
    )
    if len(mats) == 0:
        raise ValueError(f"No *_label.mat files found in: {gt_root}")
    return [os.path.join(gt_root, f) for f in mats]

def infer_labels_from_mat(mat_path: str, n_frames: int) -> np.ndarray:
    """
    volLabel: (1,T) object array, each entry is (H,W) mask.
    label[t]=1 if any pixel > 0 else 0
    """
    mat = loadmat(mat_path)
    if "volLabel" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"'volLabel' not found in {mat_path}. Keys={keys}")

    v = mat["volLabel"].reshape(-1)  # (T,)
    labels = np.zeros(len(v), dtype=np.int64)
    for i in range(len(v)):
        mask = np.array(v[i])
        labels[i] = 1 if np.any(mask > 0) else 0

    if len(labels) != n_frames:
        labels = labels[:min(len(labels), n_frames)]
    return labels

def build_per_video_gt(test_video_folders: List[str], gt_root: str, frames_test_dir: str) -> Dict[str, np.ndarray]:
    mat_paths = find_label_mat_files(gt_root)
    if len(mat_paths) < len(test_video_folders):
        raise ValueError(f"Not enough GT mats. gt={len(mat_paths)} test_videos={len(test_video_folders)}")

    per_video = {}
    for i, vname in enumerate(test_video_folders):
        vdir = os.path.join(frames_test_dir, vname)
        frames = sorted(
            [f for f in os.listdir(vdir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))],
            key=natural_key
        )
        n_frames = len(frames)
        per_video[vname] = infer_labels_from_mat(mat_paths[i], n_frames)

    return per_video


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = f"{MODEL_NAME}_T{SEQUENCE_LENGTH}_S{IMAGE_SIZE}_bs{BATCH_SIZE}_lr{LR}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    ensure_dir(run_dir)

    print("CWD:", os.getcwd())
    print("Run dir:", run_dir)

    # ----------------------------
    # TRAIN
    # ----------------------------
    print("\n===== TRAINING =====")
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

    model = FutureFramePredictorBaseline().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_losses = []
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
    ensure_dir(os.path.dirname(model_path))
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
        "scoring": "pixel_only_mse"
    })

    # ----------------------------
    # TEST (PIXEL MSE SCORES)
    # ----------------------------
    print("\n===== TESTING (Pixel-only MSE) =====")
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

    eval_model = FutureFramePredictorBaseline().to(device)
    eval_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    eval_model.eval()

    scores_raw = []
    meta = []  # (video_name, frame_idx)

    with torch.no_grad():
        for inputs, target, video_name, target_idx in tqdm(test_loader, desc="Scoring"):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = eval_model(inputs)
            pixel_err = torch.mean((pred - target) ** 2)

            scores_raw.append(pixel_err.item())
            meta.append((video_name[0], int(target_idx.item())))

    scores = normalize_scores(scores_raw)

    # Save score curve early
    plt.figure(figsize=(12, 4))
    plt.plot(scores)
    plt.title("Normalized Pixel MSE Scores (Baseline)")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "score_curve.png"), dpi=150)
    plt.close()

    # ----------------------------
    # AUC
    # ----------------------------
    print("\n===== AUC =====")
    frames_test_dir = os.path.join(FRAMES_ROOT, "test")
    test_video_folders = test_dataset.video_folders

    per_video_gt = build_per_video_gt(
        test_video_folders=test_video_folders,
        gt_root=GT_ROOT,
        frames_test_dir=frames_test_dir
    )

    gt_labels = []
    for (vname, frame_idx) in meta:
        labels = per_video_gt[vname]
        if frame_idx < 0 or frame_idx >= len(labels):
            raise ValueError(f"Frame idx out of range for {vname}: idx={frame_idx}, len={len(labels)}")
        gt_labels.append(int(labels[frame_idx]))

    gt_labels = np.array(gt_labels, dtype=np.int64)

    auc_value = roc_auc_score(gt_labels, scores)
    print(f"Avenue AUC (Baseline Pixel-only): {auc_value:.4f}")

    fpr, tpr, _ = roc_curve(gt_labels, scores)
    save_test_artifacts(run_dir, scores, gt_labels, auc_value, fpr, tpr)

    # ----------------------------
    # Qualitative sample
    # ----------------------------
    idx = min(200, len(test_dataset) - 1)
    with torch.no_grad():
        x, y, vname, fidx = test_dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        pred = eval_model(x)

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
