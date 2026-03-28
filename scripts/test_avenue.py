# ============================================================
# test_avenue_dump_all.py
#
# Avenue - TEST ONLY (Baseline + Attention)
# Dumps ALL outputs into: testing_avenue/
#
# Assumes you already extracted frames (fast):
#   ../datasets/Avenue/frames/test/<video_folder>/*.jpg|png
#
# Assumes Avenue GT mats exist (one per test video) directly under:
#   ../datasets/Avenue/test_gt/1_label.mat
#   ../datasets/Avenue/test_gt/2_label.mat
#   ...
# and each .mat contains key 'volLabel' (1, T) object array,
# where each entry is a (H,W) mask for that frame.
#
# Outputs:
# 1) Baseline (pixel-only):
#    - baseline_mse_curve.png
#    - baseline_roc_curve.png
#    - baseline_pred_vs_gt.png
#    - baseline_errors.npy
#
# 2) Attention (pixel-only):
#    - attention_mse_curve.png
#    - attention_roc_curve_pixel.png
#    - attention_pred_vs_gt_pixel.png
#    - attention_errors.npy
#
# 3) Attention (pixel+feature):
#    - attention_pixel_curve.png
#    - attention_feature_curve.png
#    - attention_combined_curve.png
#    - attention_roc_curve_pixelfeature.png
#    - attention_pred_vs_gt_pixelfeature.png
#    - attention_pixel_errors.npy
#    - attention_feature_errors.npy
#    - attention_combined_errors.npy
#
# 4) Comparisons:
#    - roc_baseline_vs_attention_pixel.png
#    - roc_baseline_vs_attention_pixelfeature.png
#
# Notes:
# - GT is aligned by dropping first SEQUENCE_LENGTH frames per test video,
#   because predictions start at target frame index = SEQUENCE_LENGTH.
# - Predictions are computed in dataset order, so lengths must match exactly.
# - Windows safe with num_workers > 0 (everything in main()).
# ============================================================

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.io import loadmat


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = "../datasets/Avenue"
FRAMES_TEST_DIR = os.path.join(ROOT_DIR, "frames", "test")
GT_ROOT = os.path.join(ROOT_DIR, "test_gt")  # contains 1_label.mat, 2_label.mat, ...

SEQUENCE_LENGTH = 5
IMAGE_SIZE = 128

NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoints (edit)
BASELINE_CKPT = "baseline_Avenue.pth"        # your baseline Avenue checkpoint
ATTENTION_CKPT = "attention_Avenue.pth"      # your attention Avenue checkpoint

# Combined score weights (attention pixel+feature)
ALPHA = 0.3
BETA = 0.7

# Dump folder
OUT_DIR = "testing_avenue"


# ============================================================
# UTILS
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def save_curve(y: np.ndarray, path: str, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(12, 4))
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_roc_curve(gt: np.ndarray, scores: np.ndarray, path: str, title: str, label: str):
    fpr, tpr, _ = roc_curve(gt, scores)
    auc_val = roc_auc_score(gt, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return fpr, tpr, auc_val

def save_roc_compare(gt: np.ndarray, score1: np.ndarray, score2: np.ndarray,
                     label1: str, label2: str, path: str, title: str):
    fpr1, tpr1, _ = roc_curve(gt, score1)
    fpr2, tpr2, _ = roc_curve(gt, score2)
    auc1 = roc_auc_score(gt, score1)
    auc2 = roc_auc_score(gt, score2)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, label=f"{label1} (AUC={auc1:.3f})")
    plt.plot(fpr2, tpr2, label=f"{label2} (AUC={auc2:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return auc1, auc2

def save_pred_vs_gt(gt_img: np.ndarray, pred_img: np.ndarray, path: str, title_suffix: str = ""):
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
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# AVENUE GT LOADING (volLabel)
# ============================================================
def find_label_mat_files(gt_root: str):
    """
    Looks for files like: 1_label.mat, 2_label.mat, ...
    """
    mats = [f for f in os.listdir(gt_root) if f.lower().endswith(".mat") and "_label" in f.lower()]
    if len(mats) == 0:
        raise ValueError(f"No *_label.mat files found in {gt_root}")
    mats = sorted(mats, key=natural_key)
    return [os.path.join(gt_root, f) for f in mats]

def infer_labels_from_mat_volLabel(mat_path: str) -> np.ndarray:
    """
    Avenue GT (.mat): key 'volLabel' is (1, T) object array.
    Each entry is a (H, W) mask. Anomaly if any pixel > 0.
    Returns labels (T,) in {0,1}
    """
    mat = loadmat(mat_path)
    if "volLabel" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"'volLabel' not found in {mat_path}. Keys={keys}")

    v = mat["volLabel"].reshape(-1)  # T objects
    labels = np.zeros(len(v), dtype=np.int64)

    for i in range(len(v)):
        mask = np.array(v[i])
        labels[i] = 1 if np.any(mask > 0) else 0

    return labels

def load_avenue_gt_aligned(frames_test_dir: str, gt_root: str, seq_len: int) -> np.ndarray:
    """
    Builds a single concatenated GT label vector aligned with predictions.
    It aligns per video by dropping first seq_len frames (no predictions for them).
    Matching is done by sorted order: test video folders <-> sorted *_label.mat files.
    """
    test_videos = sorted([d for d in os.listdir(frames_test_dir) if os.path.isdir(os.path.join(frames_test_dir, d))],
                         key=natural_key)
    mat_paths = find_label_mat_files(gt_root)

    if len(mat_paths) < len(test_videos):
        raise ValueError(f"Not enough GT mats: gt={len(mat_paths)} test_videos={len(test_videos)}")

    all_labels = []
    for i, vname in enumerate(test_videos):
        vdir = os.path.join(frames_test_dir, vname)
        frames = sorted([f for f in os.listdir(vdir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))],
                        key=natural_key)
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError(f"No frames found in: {vdir}")

        labels = infer_labels_from_mat_volLabel(mat_paths[i])

        # Align to frame count (safety)
        m = min(len(labels), n_frames)
        labels = labels[:m]

        # Drop first seq_len frames for prediction alignment
        if m <= seq_len:
            continue
        labels = labels[seq_len:]

        all_labels.extend(labels.tolist())

    return np.array(all_labels, dtype=np.int64)


# ============================================================
# DATASET (frames) - aligned indexing like Ped2 test script
# ============================================================
class AvenueFramesTestDataset(Dataset):
    """
    Expected:
      FRAMES_TEST_DIR/<video_folder>/*.jpg|png

    Returns:
      x: (T,1,H,W)
      y: (1,H,W)
      global_aligned_index: int (index into aligned GT array)
    """
    def __init__(self, frames_test_dir: str, sequence_length=5, image_size=128):
        self.frames_test_dir = frames_test_dir
        self.sequence_length = sequence_length
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.samples = []  # (video_path, frames, start, global_aligned_index)
        self._prepare_samples()

    def _prepare_samples(self):
        video_folders = sorted(
            [d for d in os.listdir(self.frames_test_dir) if os.path.isdir(os.path.join(self.frames_test_dir, d))],
            key=natural_key
        )

        global_idx = 0
        for video in video_folders:
            vpath = os.path.join(self.frames_test_dir, video)
            frames = sorted(
                [f for f in os.listdir(vpath) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))],
                key=natural_key
            )

            # Predictions exist for target indices seq_len..N-1 => N-seq_len samples
            for start in range(len(frames) - self.sequence_length):
                # This sample corresponds to aligned GT index = global_idx
                self.samples.append((vpath, frames, start, global_idx))
                global_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vpath, frames, start, gidx = self.samples[idx]

        xs = []
        for t in range(self.sequence_length):
            img = Image.open(os.path.join(vpath, frames[start + t])).convert("L")
            xs.append(self.transform(img))
        x = torch.stack(xs, dim=0)

        target_idx = start + self.sequence_length
        y = Image.open(os.path.join(vpath, frames[target_idx])).convert("L")
        y = self.transform(y)

        return x, y, int(gidx)


# ============================================================
# MODELS (same as Ped2)
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

class FutureFramePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.convlstm = ConvLSTM(128, 128)
        self.decoder = Decoder()
    def forward(self, x):
        enc = torch.stack([self.encoder(x[:, t]) for t in range(x.size(1))], dim=1)
        h = self.convlstm(enc)
        return self.decoder(h)

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
# TEST ROUTINES
# ============================================================
def test_pixel_only(model: nn.Module, loader: DataLoader, device: torch.device, pick_index=200):
    model.eval()
    errs = np.zeros(len(loader.dataset), dtype=np.float64)

    sample_gt = None
    sample_pred = None

    with torch.no_grad():
        for x, y, gidx in tqdm(loader, desc="Testing (pixel-only)"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            mse = torch.mean((pred - y) ** 2).item()
            errs[gidx] = mse

            if sample_gt is None and gidx == pick_index:
                sample_gt = y[0, 0].detach().cpu().numpy()
                sample_pred = pred[0, 0].detach().cpu().numpy()

    return errs, sample_gt, sample_pred

def test_attention_pixel_feature(model: nn.Module, loader: DataLoader, device: torch.device, alpha: float, beta: float, pick_index=200):
    model.eval()
    pixel_errs = np.zeros(len(loader.dataset), dtype=np.float64)
    feat_errs = np.zeros(len(loader.dataset), dtype=np.float64)
    comb_errs = np.zeros(len(loader.dataset), dtype=np.float64)

    sample_gt = None
    sample_pred = None

    with torch.no_grad():
        for x, y, gidx in tqdm(loader, desc="Testing (pixel+feature)"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)

            pixel = torch.mean((pred - y) ** 2)

            feat_pred = model.encoder(pred)
            feat_gt = model.encoder(y)
            feat = torch.mean((feat_pred - feat_gt) ** 2)

            comb = alpha * pixel + beta * feat

            pixel_errs[gidx] = float(pixel.item())
            feat_errs[gidx] = float(feat.item())
            comb_errs[gidx] = float(comb.item())

            if sample_gt is None and gidx == pick_index:
                sample_gt = y[0, 0].detach().cpu().numpy()
                sample_pred = pred[0, 0].detach().cpu().numpy()

    return pixel_errs, feat_errs, comb_errs, sample_gt, sample_pred


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUT_DIR)
    print("Device:", DEVICE)
    print("Dump folder:", OUT_DIR)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # GT aligned
    gt_labels = load_avenue_gt_aligned(FRAMES_TEST_DIR, GT_ROOT, SEQUENCE_LENGTH)
    print("Aligned GT frames:", len(gt_labels))
    print("Anomalous frames:", int(gt_labels.sum()))

    # Dataset + loader
    test_dataset = AvenueFramesTestDataset(FRAMES_TEST_DIR, SEQUENCE_LENGTH, IMAGE_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    print("Prediction frames (should match GT):", len(test_dataset))
    assert len(test_dataset) == len(gt_labels), "GT labels and predictions are not aligned!"

    # --------------------------------------------------------
    # 1) BASELINE (pixel-only)
    # --------------------------------------------------------
    baseline_dir = os.path.join(OUT_DIR, "baseline_pixel_only")
    ensure_dir(baseline_dir)

    baseline = FutureFramePredictor().to(DEVICE)
    baseline.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=True))

    baseline_errors, b_gt, b_pred = test_pixel_only(baseline, test_loader, DEVICE, pick_index=200)

    np.save(os.path.join(baseline_dir, "baseline_errors.npy"), baseline_errors.astype(np.float32))

    baseline_scores = normalize(baseline_errors)
    np.save(os.path.join(baseline_dir, "baseline_scores_norm.npy"), baseline_scores.astype(np.float32))

    save_curve(baseline_errors, os.path.join(baseline_dir, "baseline_mse_curve.png"),
               "Avenue Baseline - Pixel MSE (raw)", "Frame Index", "MSE")

    b_fpr, b_tpr, b_auc = save_roc_curve(
        gt_labels, baseline_scores,
        os.path.join(baseline_dir, "baseline_roc_curve.png"),
        "Avenue Baseline ROC (Pixel MSE)", "Baseline"
    )
    print(f"Baseline AUC (MSE): {b_auc:.4f}")

    if b_gt is not None:
        save_pred_vs_gt(b_gt, b_pred, os.path.join(baseline_dir, "baseline_pred_vs_gt.png"), " (idx=200)")

    # --------------------------------------------------------
    # 2) ATTENTION (pixel-only)
    # --------------------------------------------------------
    attn_pixel_dir = os.path.join(OUT_DIR, "attention_pixel_only")
    ensure_dir(attn_pixel_dir)

    attn = FutureFramePredictorWithAttention().to(DEVICE)
    attn.load_state_dict(torch.load(ATTENTION_CKPT, map_location=DEVICE, weights_only=True))

    attn_errors, a_gt, a_pred = test_pixel_only(attn, test_loader, DEVICE, pick_index=200)

    np.save(os.path.join(attn_pixel_dir, "attention_errors.npy"), attn_errors.astype(np.float32))
    attn_scores = normalize(attn_errors)
    np.save(os.path.join(attn_pixel_dir, "attention_scores_norm.npy"), attn_scores.astype(np.float32))

    save_curve(attn_errors, os.path.join(attn_pixel_dir, "attention_mse_curve.png"),
               "Avenue Attention - Pixel MSE (raw)", "Frame Index", "MSE")

    a_fpr, a_tpr, a_auc = save_roc_curve(
        gt_labels, attn_scores,
        os.path.join(attn_pixel_dir, "attention_roc_curve_pixel.png"),
        "Avenue Attention ROC (Pixel MSE)", "Attention"
    )
    print(f"Attention AUC (Pixel MSE): {a_auc:.4f}")

    if a_gt is not None:
        save_pred_vs_gt(a_gt, a_pred, os.path.join(attn_pixel_dir, "attention_pred_vs_gt_pixel.png"), " (idx=200)")

    # --------------------------------------------------------
    # 3) ATTENTION (pixel+feature)
    # --------------------------------------------------------
    attn_pf_dir = os.path.join(OUT_DIR, "attention_pixel_feature")
    ensure_dir(attn_pf_dir)

    pixel_errs, feat_errs, comb_errs, ap_gt, ap_pred = test_attention_pixel_feature(
        attn, test_loader, DEVICE, ALPHA, BETA, pick_index=200
    )

    np.save(os.path.join(attn_pf_dir, "attention_pixel_errors.npy"), pixel_errs.astype(np.float32))
    np.save(os.path.join(attn_pf_dir, "attention_feature_errors.npy"), feat_errs.astype(np.float32))
    np.save(os.path.join(attn_pf_dir, "attention_combined_errors.npy"), comb_errs.astype(np.float32))

    pixel_scores = normalize(pixel_errs)
    feat_scores = normalize(feat_errs)
    comb_scores = normalize(comb_errs)

    np.save(os.path.join(attn_pf_dir, "pixel_scores_norm.npy"), pixel_scores.astype(np.float32))
    np.save(os.path.join(attn_pf_dir, "feature_scores_norm.npy"), feat_scores.astype(np.float32))
    np.save(os.path.join(attn_pf_dir, "combined_scores_norm.npy"), comb_scores.astype(np.float32))

    save_curve(pixel_errs, os.path.join(attn_pf_dir, "attention_pixel_curve.png"),
               "Avenue Attention - Pixel Error (raw)", "Frame Index", "Pixel MSE")
    save_curve(feat_errs, os.path.join(attn_pf_dir, "attention_feature_curve.png"),
               "Avenue Attention - Feature Error (raw)", "Frame Index", "Feature MSE")
    save_curve(comb_errs, os.path.join(attn_pf_dir, "attention_combined_curve.png"),
               "Avenue Attention - Combined Score (raw)", "Frame Index", "Score")

    ap_fpr, ap_tpr, ap_auc = save_roc_curve(
        gt_labels, comb_scores,
        os.path.join(attn_pf_dir, "attention_roc_curve_pixelfeature.png"),
        "Avenue Attention ROC (Pixel+Feature Combined)", "Attention (Pixel+Feature)"
    )
    print(f"Attention AUC (Pixel+Feature): {ap_auc:.4f}")

    if ap_gt is not None:
        save_pred_vs_gt(ap_gt, ap_pred, os.path.join(attn_pf_dir, "attention_pred_vs_gt_pixelfeature.png"), " (idx=200)")

    # --------------------------------------------------------
    # 4) COMPARISONS
    # --------------------------------------------------------
    comp_dir = os.path.join(OUT_DIR, "comparisons")
    ensure_dir(comp_dir)

    # Baseline vs Attention (both pixel-only)
    save_roc_compare(
        gt_labels, baseline_scores, attn_scores,
        "Baseline", "Attention",
        os.path.join(comp_dir, "roc_baseline_vs_attention_pixel.png"),
        "Avenue ROC Comparison (Both Pixel MSE)"
    )

    # Baseline vs Attention (baseline pixel-only vs attention pixel+feature)
    save_roc_compare(
        gt_labels, baseline_scores, comb_scores,
        "Baseline (Pixel)", "Attention (Pixel+Feature)",
        os.path.join(comp_dir, "roc_baseline_vs_attention_pixelfeature.png"),
        "Avenue ROC Comparison (Baseline Pixel vs Attention Pixel+Feature)"
    )

    # Summary
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Baseline AUC (Pixel MSE): {b_auc:.6f}\n")
        f.write(f"Attention AUC (Pixel MSE): {a_auc:.6f}\n")
        f.write(f"Attention AUC (Pixel+Feature): {ap_auc:.6f}\n")
        f.write(f"Aligned GT frames: {len(gt_labels)} | Anomalous: {int(gt_labels.sum())}\n")
        f.write(f"Checkpoints: baseline={BASELINE_CKPT}, attention={ATTENTION_CKPT}\n")

    print("\nDone. All outputs dumped to:", OUT_DIR)


if __name__ == "__main__":
    main()
