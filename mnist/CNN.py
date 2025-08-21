import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import struct
from array import array


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
    images = np.expand_dims(images, axis=1)


    dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=shuffle)
    return dataloader

train_data = get_dataloader(training_images_filepath, training_labels_filepath, shuffle=True)
test_data = get_dataloader(test_images_filepath, test_labels_filepath, shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class CNN(nn.Module):
    def __init__(self):
        """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
        super(CNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # (batch, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                  # (batch, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),# (batch, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                                   # (batch, 64, 7, 7)
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)        # keep spatial structure
        x = torch.flatten(x, 1)       # flatten to (batch, 64*7*7)
        logits = self.fc_stack(x)
        return logits

model = CNN().to(device)
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