# ============================================================
# train_test_ped2_baseline_pixelonly.py
#
# UCSD Ped2 - Baseline Future Frame Prediction (NO ATTENTION)
# Scoring: Pixel MSE only
#
# Safe for Windows with num_workers > 0:
# - Dataset/model/classes are defined at top-level
# - Training/testing are inside main()
# - Uses if __name__ == "__main__": main()
# ============================================================

import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = "../datasets/UCSDped2"
SEQUENCE_LENGTH = 5
IMAGE_SIZE = 128

BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 50

# Set >0 to enable multiprocessing dataloading
NUM_WORKERS_TRAIN = 2
NUM_WORKERS_TEST = 2

# Output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
RUN_NAME = f"ped2_baseline_pixelonly_T{SEQUENCE_LENGTH}_S{IMAGE_SIZE}_bs{BATCH_SIZE}_lr{LR}_ep{NUM_EPOCHS}"
OUT_DIR = os.path.join(RUNS_DIR, RUN_NAME)

MODEL_PATH = os.path.join(OUT_DIR, "baseline_ucsd_ped2.pth")


# ============================================================
# UTILS
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_loss_curve(losses: List[float], out_path: str, title: str):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_error_curve(errors: List[float], out_path: str, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(errors)
    plt.title(title)
    plt.xlabel("Clip Index")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_pred_vs_gt(gt_img: np.ndarray, pred_img: np.ndarray, out_path: str, title_suffix: str = ""):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth" + title_suffix)
    plt.imshow(gt_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted" + title_suffix)
    plt.imshow(pred_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# DATASET
# ============================================================
class UCSDPed2Dataset(Dataset):
    """
    Expects:
      ROOT_DIR/train/<video_folder>/*.jpg|png|tif
      ROOT_DIR/test/<video_folder>/*.jpg|png|tif

    Returns:
      input_tensor: (T, 1, H, W)
      target_img:   (1, H, W)
    """
    def __init__(self, root_dir, sequence_length=5, image_size=128, mode="train"):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.mode = mode

        self.video_dir = os.path.join(root_dir, "train" if mode == "train" else "test")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.samples: List[Tuple[str, List[str], str]] = []
        self._prepare_samples()

    def _prepare_samples(self):
        video_folders = sorted(os.listdir(self.video_dir))

        for video in video_folders:
            video_path = os.path.join(self.video_dir, video)

            if not os.path.isdir(video_path):
                continue

            frames = sorted([
                f for f in os.listdir(video_path)
                if f.lower().endswith((".jpg", ".png", ".tif"))
            ])

            # create sequences: input [i..i+T-1], target i+T
            for i in range(len(frames) - self.sequence_length):
                self.samples.append((
                    video_path,
                    frames[i:i + self.sequence_length],
                    frames[i + self.sequence_length]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, input_frames, target_frame = self.samples[idx]

        xs = []
        for f in input_frames:
            img = Image.open(os.path.join(video_path, f)).convert("L")
            img = self.transform(img)
            xs.append(img)

        input_tensor = torch.stack(xs, dim=0)  # (T, 1, H, W)

        target_img = Image.open(os.path.join(video_path, target_frame)).convert("L")
        target_img = self.transform(target_img)  # (1, H, W)

        return input_tensor, target_img


# ============================================================
# MODEL (Baseline)
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # 32 -> 16
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
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
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


class FutureFramePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.convlstm = ConvLSTM(128, 128)
        self.decoder = Decoder()

    def forward(self, x):
        # x: (B, T, 1, H, W)
        encoded = torch.stack([self.encoder(x[:, t]) for t in range(x.size(1))], dim=1)
        h = self.convlstm(encoded)
        return self.decoder(h)


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Output dir:", OUT_DIR)

    # Slight speedup on CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ----------------------------
    # DATA
    # ----------------------------
    train_dataset = UCSDPed2Dataset(
        root_dir=ROOT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE,
        mode="train"
    )

    test_dataset = UCSDPed2Dataset(
        root_dir=ROOT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        image_size=IMAGE_SIZE,
        mode="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS_TRAIN > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS_TEST,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS_TEST > 0)
    )

    # ----------------------------
    # MODEL
    # ----------------------------
    model = FutureFramePredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ----------------------------
    # TRAIN
    # ----------------------------
    print("\n===== TRAINING =====")
    epoch_losses = []
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, target in pbar:
            inputs = inputs.to(device, non_blocking=True)   # (B, T, 1, H, W)
            target = target.to(device, non_blocking=True)   # (B, 1, H, W)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    save_loss_curve(
        epoch_losses,
        os.path.join(OUT_DIR, "loss_curve.png"),
        title="UCSD Ped2 Baseline Training Loss (Pixel-only)"
    )

    print(f"Training time: {time.time() - t0:.1f} sec")

    # ----------------------------
    # TEST (Pixel MSE list)
    # ----------------------------
    print("\n===== TESTING (Pixel MSE) =====")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    errors = []
    with torch.no_grad():
        for inputs, target in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(inputs)
            mse = torch.mean((pred - target) ** 2).item()
            errors.append(mse)

    np.save(os.path.join(OUT_DIR, "pixel_mse_errors.npy"), np.array(errors, dtype=np.float32))
    save_error_curve(
        errors,
        os.path.join(OUT_DIR, "mse_curve.png"),
        title="UCSD Ped2 Test Prediction Error (Pixel MSE)"
    )

    # ----------------------------
    # QUALITATIVE SAMPLE
    # ----------------------------
    sample_idx = min(200, len(test_dataset) - 1)
    with torch.no_grad():
        x, y = test_dataset[sample_idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        pred = model(x)

        save_pred_vs_gt(
            gt_img=y[0, 0].cpu().numpy(),
            pred_img=pred[0, 0].cpu().numpy(),
            out_path=os.path.join(OUT_DIR, f"sample_{sample_idx}_pred_vs_gt.png"),
            title_suffix=f"\n(sample idx={sample_idx})"
        )

    print("\nDone.")
    print("Artifacts saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
