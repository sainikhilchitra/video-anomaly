import os, re
import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT_DIR = "../datasets/Avenue"
FRAMES_ROOT = os.path.join(ROOT_DIR, "frames")
GT_ROOT = os.path.join(ROOT_DIR, "test_gt")

MODEL_PATH = "attention_Avenue.pth"   # <-- change if needed
SEQUENCE_LENGTH = 5
IMAGE_SIZE = 128
ALPHA = 0.3
BETA = 0.7
NUM_WORKERS = 2

# -------------- HELPERS -----------------
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def normalize_scores(x):
    x = np.array(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def list_gt_matfiles(gt_dir):
    mats = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".mat") and "_label" in f.lower()],
                  key=natural_key)
    if not mats:
        raise ValueError(f"No *_label.mat found in {gt_dir}")
    return [os.path.join(gt_dir, f) for f in mats]

def infer_labels_from_mat(mat_path: str, n_frames: int):
    mat = loadmat(mat_path)
    v = mat["volLabel"].reshape(-1)  # (T,)
    labels = np.zeros(len(v), dtype=np.int64)
    for i in range(len(v)):
        mask = np.array(v[i])
        labels[i] = 1 if np.any(mask > 0) else 0
    if len(labels) != n_frames:
        labels = labels[:min(len(labels), n_frames)]
    return labels

# -------------- DATASET -----------------
class AvenueFramesDataset(Dataset):
    def __init__(self, frames_root, sequence_length=5, image_size=128, mode="test"):
        self.sequence_length = sequence_length
        self.video_dir = os.path.join(frames_root, "test" if mode == "test" else "train")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.samples = []
        self.video_folders = []
        self._prepare()

    def _prepare(self):
        self.video_folders = sorted(
            [d for d in os.listdir(self.video_dir) if os.path.isdir(os.path.join(self.video_dir, d))],
            key=natural_key
        )
        for v in self.video_folders:
            vp = os.path.join(self.video_dir, v)
            frames = sorted([f for f in os.listdir(vp) if f.lower().endswith((".jpg",".png",".tif",".jpeg"))],
                            key=natural_key)
            for start in range(len(frames) - self.sequence_length):
                self.samples.append((vp, v, frames, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, vname, frames, start = self.samples[idx]
        xs = []
        for t in range(self.sequence_length):
            img = Image.open(os.path.join(vp, frames[start+t])).convert("L")
            xs.append(self.transform(img))
        x = torch.stack(xs, dim=0)
        tidx = start + self.sequence_length
        y = Image.open(os.path.join(vp, frames[tidx])).convert("L")
        y = self.transform(y)
        return x, y, vname, tidx

# -------------- MODEL -------------------
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
    def forward(self, x): return self.encoder(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4*hidden_dim, 3, padding=1)
    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x,h], dim=1))
        i,f,o,g = torch.split(gates, self.hidden_dim, dim=1)
        i,f,o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h,c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
    def forward(self, x):
        B,T,C,H,W = x.size()
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h,c = self.cell(x[:,t], h, c)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(x)

class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c//reduction, 1, bias=False), nn.ReLU(),
            nn.Conv2d(c//reduction, c, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        return self.sig(self.conv(torch.cat([avg,mx], dim=1)))

class SpatioTemporalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class FutureFramePredictorWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.convlstm = ConvLSTM(128, 128)
        self.att = SpatioTemporalAttention(128)
        self.decoder = Decoder()
    def forward(self, x):
        enc = torch.stack([self.encoder(x[:,t]) for t in range(x.size(1))], dim=1)
        h = self.convlstm(enc)
        h = self.att(h)
        return self.decoder(h)

# -------------- AUC RUN -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_ds = AvenueFramesDataset(FRAMES_ROOT, sequence_length=SEQUENCE_LENGTH, image_size=IMAGE_SIZE, mode="test")
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True, persistent_workers=(NUM_WORKERS>0))

    model = FutureFramePredictorWithAttention().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    scores = []
    meta = []

    with torch.no_grad():
        for x, y, vname, tidx in tqdm(test_ld, desc="Scoring"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)

            pix = torch.mean((pred - y) ** 2)
            f_pred = model.encoder(pred)
            f_gt = model.encoder(y)
            feat = torch.mean((f_pred - f_gt) ** 2)

            s = ALPHA * pix + BETA * feat
            scores.append(s.item())
            meta.append((vname[0], int(tidx.item())))

    scores = normalize_scores(scores)

    # Build GT map per video
    gt_files = list_gt_matfiles(GT_ROOT)
    frames_test_dir = os.path.join(FRAMES_ROOT, "test")

    per_video_gt = {}
    for i, vfolder in enumerate(test_ds.video_folders):
        vdir = os.path.join(frames_test_dir, vfolder)
        n_frames = len([f for f in os.listdir(vdir) if f.lower().endswith((".jpg",".png",".tif",".jpeg"))])
        per_video_gt[vfolder] = infer_labels_from_mat(gt_files[i], n_frames)

    gt = np.array([per_video_gt[v][f] for (v,f) in meta], dtype=np.int64)

    auc = roc_auc_score(gt, scores)
    print("Avenue AUC:", round(float(auc), 4))

    fpr, tpr, _ = roc_curve(gt, scores)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.title("Avenue ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.grid(True); plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
