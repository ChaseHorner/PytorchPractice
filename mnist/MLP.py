import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


import numpy as np
import struct
from array import array
from os.path  import join


# MNIST Data Load
training_images_filepath = r'mnist\train-images-idx3-ubyte\train-images-idx3-ubyte'
training_labels_filepath = r'mnist\train-labels-idx1-ubyte\train-labels-idx1-ubyte'
test_images_filepath = r'mnist\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte'
test_labels_filepath = r'mnist\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte'

def get_dataloader(images_filepath, labels_filepath, shuffle = False):
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
        image_data = array("B", file.read())

    images = np.frombuffer(image_data, dtype=np.uint8).reshape(size, rows, cols)
    images = images.astype(np.float32) / 255.0

    dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=shuffle)
    return dataloader

train_data = get_dataloader(training_images_filepath, training_labels_filepath, shuffle=True)
test_data = get_dataloader(test_images_filepath, test_labels_filepath, shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


def train(train_data, model, loss_fn, optimizer):
    size = len(train_data.dataset)
    model.train()
    for batch, (X, y), in enumerate(train_data):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            accuracy = (pred.argmax(1) == y).type(torch.float).sum().item() / len(X) * 100
            print(f"accuracy: {accuracy:>0.1f}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(test_data, model, loss_fn):
    size = len(test_data.dataset)
    num_batches = len(test_data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



#Training and Testing
epochs = 5
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data, model, loss_fn, optimizer)
    test(test_data, model, loss_fn)
print("Done!")