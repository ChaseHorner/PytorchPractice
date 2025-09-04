import torch
import matplotlib.pyplot as plt
import numpy as np

import os
from dataset import ImageDataset
from model import SR_Unet
from collections import OrderedDict
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(66)

def generate_images(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)
    inputs, labels, predictions = inputs.cpu().numpy(), labels.cpu().numpy(), predictions.cpu().numpy()
    plt.figure(figsize=(15,20))

    display_list = [inputs[-1].transpose((1, 2, 0)), labels[-1].transpose((1, 2, 0)), predictions[-1].transpose((1, 2, 0))]
    title = ['Input', 'Real', 'Predicted']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        img = display_list[i]
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)
        
        plt.imshow(img)    
        plt.axis('off')
    plt.show()


def generate_random_images(model, inputs, labels, num_images=3):
    model.eval()
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)
    
    # move to CPU and convert to numpy
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    batch_size = inputs.shape[0]
    num_images = min(num_images, batch_size)
    random_indices = np.random.choice(batch_size, num_images, replace=False)
    
    for idx in random_indices:
        plt.figure(figsize=(12, 4))
        display_list = [
            inputs[idx].transpose(1, 2, 0),
            predictions[idx].transpose(1, 2, 0),
            labels[idx].transpose(1, 2, 0)
        ]
        titles = ['Input', 'Prediction', 'Target']
        
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(titles[i])

            img = display_list[i]
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min)
            
            plt.imshow(img)    
            plt.axis('off')
        
        output_path = os.path.join("UNET", f"example_{idx}")
        plt.savefig(output_path)



if __name__ == "__main__":
    torch.manual_seed(66)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Paths
    TRAIN_DATA_PATH = os.path.join('data', 'train')
    VAL_DATA_PATH = os.path.join('data', 'val')
    MODEL_PATH = os.path.join('UNET', 'SR_unet_model_1.pt')

    # DataLoaders
    val_dataset = ImageDataset(VAL_DATA_PATH, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    inputs, labels = next(iter(val_loader))

    # Initialize the model architecture
    SR_unet_model = SR_Unet(n_channels=3, n_classes=3).to(device)

    # Load the saved weights, handling potential DataParallel 'module.' prefix
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        state_dict = new_state_dict

    SR_unet_model.load_state_dict(state_dict)
    SR_unet_model.eval()  # Set model to evaluation mode

    # Generate random images
    generate_random_images(SR_unet_model, inputs.to(device), labels.to(device), num_images=10)