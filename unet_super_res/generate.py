import torch
import matplotlib.pyplot as plt
import numpy as np

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
        plt.imshow((display_list[i] + 1) / 2)
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
            labels[idx].transpose(1, 2, 0),
            predictions[idx].transpose(1, 2, 0)
        ]
        titles = ['Input', 'Ground Truth', 'Prediction']
        
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(titles[i])
            plt.imshow((display_list[i] + 1) / 2)  # scale if images were normalized [-1,1]
            plt.axis('off')
        
        plt.show()