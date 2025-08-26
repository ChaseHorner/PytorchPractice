from dataset import SegmentationDataset
from model import UNet
import config

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
from imutils import paths
import matplotlib.pyplot as plt
import torch
import time
import os

def main():
    imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths,
        test_size=config.TEST_SPLIT, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]


    # define transformations
    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transform)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transform)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, drop_last=False)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, drop_last=False)

    # initialize our UNet model
    unet = UNet().to(config.DEVICE)

    # initialize loss function and optimizer
    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([2]))
    optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

    #initialization
    num_epochs=40
    loss_hist=[0]*(num_epochs) #loss function history for training data
    loss_hist_test=[0]*(num_epochs) #loss function history for test data
    accuracy_hist=[0]*(num_epochs) #accuracy history for training data
    accuracy_hist_test=[0]*(num_epochs) #accuracy history for test data


    test_di=iter(testLoader)
    test_dat=next(test_di)
    torch.sum(test_dat[1]==0)+torch.sum(test_dat[1]==1)
    x_test=test_dat[0]
    y_test=test_dat[1]
    y_test[1,:,:]
    torch.unique(y_test)

    for epoch in range(num_epochs):

        if epoch==0:
            print("epoch: ",epoch,sep="",end="")
        else:
            print("")
            print("training loss: ", round(loss_hist[epoch - 1], 2),
                "; test loss: ", round(loss_hist_test[epoch - 1], 2), sep="")
            print("training accuracy: ",round(accuracy_hist[epoch-1],2),
                "; test accuracy: ",round(accuracy_hist_test[epoch-1],2),sep="")
            print("epoch: ",epoch,sep="",end="")

        unet.train()
        for x_batch,y_batch in trainLoader:
            print(".",end="")

            #make a prediction and get the loss
            optimizer.zero_grad()
            pred=unet(x_batch)
            loss = loss_fn(pred, y_batch.float())

            #gradients and then optimizer
            loss.backward()
            optimizer.step()

            #update history information on training data for this epoch
            loss_hist[epoch]+=loss.item()*y_batch.size(0)
            pred_labels=(torch.sigmoid(pred)>0.5).float()
            correct=(pred_labels.squeeze(1)==y_batch).float().sum()
            accuracy_hist[epoch]+=correct.item()

        #normalize loss and accuracy data for the training data for the epoch
        loss_hist[epoch] /= len(trainLoader.dataset)
        accuracy_hist[epoch] /= (len(trainLoader.dataset)*y_batch.shape[-1]*y_batch.shape[-2])

        #now get loss and accuracy data for the test
        unet.eval()
        with torch.no_grad():
            pred_test=unet(x_test)
            loss_hist_test[epoch]=loss_fn(pred_test,y_test).item()
            pred_test_labels = (torch.sigmoid(pred_test) > 0.5).float()
            accuracy_hist_test[epoch] = (pred_test_labels == y_test).float().mean().item()

    print("")
    print("training loss: ",round(loss_hist[epoch],2),
        "; test loss: ",round(loss_hist_test[epoch],2),sep="")
    print("training accuracy: ",round(accuracy_hist[epoch],2),
        "; test accuracy: ",round(accuracy_hist_test[epoch],2),sep="")


    #**Now make plots of results to examine some of them

    unet.eval()
    with torch.no_grad():
        pred=unet(x_test)
        pred=(torch.sigmoid(pred)>0.5).float()

        for ind in range(30):
            fig, axs=plt.subplots(1,3,figsize=(12,4))
            axs[0].imshow(x_test[ind,:,:,:].permute(1,2,0))
            axs[1].imshow(pred[ind, :, :, :].permute(1,2,0))
            axs[2].imshow(y_test[ind].squeeze())
            plt.show()


if __name__ == "__main__":
    main()
