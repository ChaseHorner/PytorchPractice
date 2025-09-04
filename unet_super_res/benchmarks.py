import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import peak_signal_noise_ratio
from dataset import ImageDataset
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure


device = torch.device('cpu')
BATCH_SIZE = 16


torch.manual_seed(66)

# Paths
TRAIN_DATA_PATH = os.path.join('data', 'train')
VAL_DATA_PATH = os.path.join('data', 'val')
BATCH_SIZE = 16
EPOCHS = 40
MODEL_NAME = 'SR_unet_model_1'

# DataLoaders
train_dataset = ImageDataset(TRAIN_DATA_PATH, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageDataset(VAL_DATA_PATH, is_train=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def compute_baseline_psnr(dataloader):
    psnrs = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        psnr_batch = peak_signal_noise_ratio(inputs, targets, data_range=1.0)
        psnrs.append(psnr_batch.mean().item())
    return sum(psnrs) / len(psnrs)

train_psnr_baseline = compute_baseline_psnr(train_loader)
val_psnr_baseline = compute_baseline_psnr(val_loader)

print(f"Baseline PSNR — Train: {train_psnr_baseline:.2f} dB, Val: {val_psnr_baseline:.2f} dB")

ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)

def compute_baseline_ssim(dataloader):
    ssims = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        ssim_batch = ssim_metric(inputs, targets)
        ssims.append(ssim_batch.mean().item())
    return sum(ssims) / len(ssims)

train_ssim_baseline = compute_baseline_ssim(train_loader)
val_ssim_baseline = compute_baseline_ssim(val_loader)

print(f"Baseline SSIM — Train: {train_ssim_baseline:.4f}, Val: {val_ssim_baseline:.4f}")
