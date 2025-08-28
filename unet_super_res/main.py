import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import os
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import ImageDataset
from model import SR_Unet
from train import train_model
from generate import generate_random_images


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR_unet_model = SR_Unet().to(device)
SR_unet_model.to(device)

criterion = nn.L1Loss()

optimizer = optim.AdamW(SR_unet_model.parameters(), lr=1e-4, betas=[0.5,0.999])

save_model = './UNET'
os.makedirs(save_model, exist_ok = True)

print(f"Training on {device} for {EPOCHS} epochs...")
SR_unet_model, metrics = train_model(
    SR_unet_model, MODEL_NAME, save_model, optimizer, criterion, train_loader, val_loader, EPOCHS, device
)

generate_random_images(SR_unet_model, val_dataset[0][0].unsqueeze(0), val_dataset[0][1].unsqueeze(0), num_images=3)

plt.figure(figsize=(30, 10))

# PSNR
plt.subplot(1, 3, 1)
plt.plot(range(1, EPOCHS + 1), metrics["train_psnr"], label='PSNR Training', color='blue', linestyle='--')
plt.plot(range(1, EPOCHS + 1), metrics["valid_psnr"], label='PSNR Validation', color='blue')
plt.xlabel('Epochs')
plt.ylabel('PSNR')
plt.legend()
plt.title('PSNR')

# SSIM
plt.subplot(1, 3, 2)
plt.plot(range(1, EPOCHS + 1), metrics["train_ssim"], label='SSIM Training', color='red', linestyle='--')
plt.plot(range(1, EPOCHS + 1), metrics["valid_ssim"], label='SSIM Validation', color='red')
plt.xlabel('Epochs')
plt.ylabel('SSIM')
plt.legend()
plt.title('SSIM')

# Loss
plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS + 1), metrics["train_loss"], label='Loss Training', color='green', linestyle='--')
plt.plot(range(1, EPOCHS + 1), metrics["valid_loss"], label='Loss Validation', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig(save_model + '/psnr_ssim_loss.png')
plt.show()
