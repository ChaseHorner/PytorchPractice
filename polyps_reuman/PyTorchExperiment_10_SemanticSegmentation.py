#This is an effort to do semantic segmentation of colon polyps based on the data in
#the SemanticSegmentationDataset folder.
#
#Reuman
#2025 06 10

import torch
import numpy as np
import pathlib
import torchvision

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

#**data and data loader

#get the file names for the images and masks
image_list=pathlib.Path("./SemanticSegmentationDataset/images")
image_list=[str(path) for path in image_list.glob("*.jpg")]
image_list=np.array(image_list,dtype=str)
mask_list=pathlib.Path("./SemanticSegmentationDataset/masks/")
mask_list=[str(path) for path in mask_list.glob("*.jpg")]
mask_list=np.array(mask_list,dtype=str)
train_image_list, test_image_list, train_mask_list, test_mask_list = train_test_split(image_list,mask_list,test_size=0.2,random_state=42)

#check out some images to see what they look like and figure out how to prep them
img=Image.open(image_list[0])
img.show()
img=transforms.Resize((100,100))(img)
img.show()

msk=Image.open(mask_list[0])
msk.show()
msk=transforms.Resize((100,100))(msk)
msk.show()
msk=transforms.ToTensor()(msk)
msk.shape #why are there 3 channels? are they all the same anyway?
torch.all(msk[0,:,:]==msk[1,:,:])
torch.all(msk[0,:,:]==msk[2,:,:]) #yes, all the same, so just take the first channel
msk=msk[0,:,:]
msk.shape
torch.sum(msk==0)+torch.sum(msk==1) #so not everything is a 0 or 1, so round
msk=torch.round(msk)
torch.sum(msk==0)+torch.sum(msk==1) #now everything is a 0 or 1

#now make a data class
class PolypData(Dataset):
    def __init__(self,image_list,mask_list):
        self.image_list=image_list
        self.mask_list=mask_list

    def __getitem__(self,index):
        #for the images, just resize and convert to tensor
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128))])
        img=Image.open(self.image_list[index])
        img=transform(img)

        #for the masks, you also have to drop from 3 to 1 channel, and round
        msk=Image.open(self.mask_list[index])
        msk=transform(msk)
        msk=msk[0,:,:]
        msk=torch.round(msk)
        return img, msk

    def __len__(self):
        return len(self.image_list)

#now make the test and train data
train_d=PolypData(image_list=train_image_list,mask_list=train_mask_list)
test_d=PolypData(image_list=test_image_list,mask_list=test_mask_list)

#examine to make sure it is what you want
d1=train_d[1]
type(d1)
len(d1)
type(d1[1])
x_batch=d1[0]
x_batch.shape
x_batch.dtype
y_batch=d1[1]
y_batch.shape
y_batch.dtype
torch.sum(d1[1]==0)+torch.sum(d1[1]==1)
128**2

#now make a data loader and iterator for the training data
train_dl=DataLoader(train_d,batch_size=16,drop_last=False,shuffle=True)
train_di=iter(train_dl)

#get the pos_weight argument you will use to the optimizer, to deal with
#unbalanced classes in outputs
tot_pos=0
tot_neg=0
for _,y_batch in train_dl:
    h=y_batch.sum().item()
    tot_pos+=h
    tot_neg+=(torch.prod(torch.tensor(y_batch.shape))-h).item()
tot_neg/(tot_pos+tot_neg) #This explains why random initial parameters lead to 85% accuracy,
#because you get that just by guessing always negative
pos_weight=tot_neg/tot_pos

#Now make a one-batch data loader and iterator for test data. There
#must be a better way to do this. Basically I just want to get the test
#data as two big tensors, one for images and one for masks.
test_dl=DataLoader(test_d,batch_size=len(test_d),drop_last=False,shuffle=False)
test_di=iter(test_dl)
test_dat=next(test_di)
type(test_dat)
len(test_dat)
test_dat[0].shape
test_dat[1].shape
torch.sum(test_dat[1]==0)+torch.sum(test_dat[1]==1)
200*128*128
x_test=test_dat[0]
y_test=test_dat[1]
type(y_test)
y_test.shape
y_test[1,:,:]
torch.unique(y_test) #so the "labels" are 128 by 128 2d tensors of 0s and 1s

#**Make a model. Drawing heavily on ideas from
#https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
#but changing them considerably as well

#Make a class for a certain block of modules that repeats in different forms
#for the u-net architecture. Does two convolutions with a relu in between,
#maintains the height and width.
class Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding="same")
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding="same")

    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))

#try it out on arbitrary tensors
b=Block(3,16)
b(torch.ones((10,3,128,128))).shape
b(torch.ones(10,3,16,16)).shape

#Make the unet module class. Influenced by
#https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
#but with some changes.
class unet(nn.Module):
    def __init__(self,channels=(3,16,32,64)):
        super().__init__()
        self.channels=channels

        self.encoder_blocks=nn.ModuleList([
            Block(channels[i],channels[i+1]) for i in range(len(channels)-2)
        ])
        self.pool=nn.MaxPool2d(2)
        self.encoder_bottom_block=Block(channels[-2],channels[-1])

        self.upconvs=nn.ModuleList([
            nn.ConvTranspose2d(in_channels=channels[-i],
                               out_channels=channels[-(i+1)],
                               kernel_size=2,stride=2)
            for i in range(1,len(channels)-1)
        ])
        self.decoder_blocks=nn.ModuleList([
            Block(2*channels[-(i+1)],channels[-(i+1)])
            for i in range(1,len(channels)-1)
        ])
        self.final_conv=nn.Conv2d(in_channels=channels[1],
                                  out_channels=1,
                                  kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        origshape=x.shape[-2:]
        blockOuts=[]

        #do the encoder
        for block in self.encoder_blocks:
            x=block(x)
            blockOuts.append(x)
            x=self.pool(x)
        x=self.encoder_bottom_block(x)

        #do the decoder
        for i in range(len(self.upconvs)):
            x=self.upconvs[i](x)
            (_, _, h, w) = x.shape
            cropped=transforms.CenterCrop([h, w])(blockOuts[-(i+1)])
            x = torch.cat([x, cropped], dim=1)
            x=self.decoder_blocks[i](x)

        #interpolate to the original height and width, and cut to one channel
        x=nn.functional.interpolate(input=x,size=origshape)
        x=self.final_conv(x)

        return x

#try it out on arbitrary tensors to see if it returns the right shape
u=unet(channels=(3,16,32,64))
u
x=torch.ones((10,3,128,128))
x.shape
res=u(x) #input a batch of 10 images, see what you get
type(res)
res.shape

#**now you can train this model in the usual way

torch.manual_seed(42)
learning_rate=0.001
pos_weight #I tried this and it resulted in bad outcome - too much was classified as polyp,
#to avoid missing anything. So try a lower value which is still bigger than 1 (the default).
#I have learned that it's a judgement call depending on what you want out of your
#segmentation solution.
loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))
model=unet(channels=(3,16,32,64,128,256))
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#initialization
num_epochs=40
loss_hist=[0]*(num_epochs) #loss function history for training data
loss_hist_test=[0]*(num_epochs) #loss function history for test data
accuracy_hist=[0]*(num_epochs) #accuracy history for training data
accuracy_hist_test=[0]*(num_epochs) #accuracy history for test data

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

    model.train()
    for x_batch,y_batch in train_dl:
        print(".",end="")

        #make a prediction and get the loss
        optimizer.zero_grad()
        pred=model(x_batch)
        loss=loss_fn(pred.squeeze(1),y_batch)

        #gradients and then optimizer
        loss.backward()
        optimizer.step()

        #update history information on training data for this epoch
        loss_hist[epoch]+=loss.item()*y_batch.size(0)
        pred_labels=(torch.sigmoid(pred)>0.5).float()
        correct=(pred_labels.squeeze(1)==y_batch).float().sum()
        accuracy_hist[epoch]+=correct.item()

    #normalize loss and accuracy data for the training data for the epoch
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= (len(train_dl.dataset)*y_batch.shape[-1]*y_batch.shape[-2])

    #now get loss and accuracy data for the test
    model.eval()
    with torch.no_grad():
        pred_test=model(x_test)
        loss_hist_test[epoch]=loss_fn(pred_test.squeeze(1),y_test).item()
        pred_test_labels = (torch.sigmoid(pred_test) > 0.5).float()
        accuracy_hist_test[epoch] = (pred_test_labels.squeeze(1) == y_test).float().mean().item()

print("")
print("training loss: ",round(loss_hist[epoch],2),
      "; test loss: ",round(loss_hist_test[epoch],2),sep="")
print("training accuracy: ",round(accuracy_hist[epoch],2),
      "; test accuracy: ",round(accuracy_hist_test[epoch],2),sep="")

#now plot the histories
plt.plot(range(num_epochs),loss_hist,linestyle="--")
plt.plot(range(num_epochs),loss_hist_test,linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Loss, solid is test data")
plt.show()

plt.plot(range(num_epochs),accuracy_hist,linestyle="--")
plt.plot(range(num_epochs),accuracy_hist_test,linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Accuracy, solid is test data")
plt.show()

#**Now make plots of results to examine some of them

model.eval()
with torch.no_grad():
    pred=model(x_test)
    pred=(torch.sigmoid(pred)>0.5).float()

    for ind in range(30):
        fig, axs=plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(x_test[ind,:,:,:].permute(1,2,0))
        axs[1].imshow(pred[ind, :, :, :].permute(1,2,0))
        axs[2].imshow(y_test[ind, :, :])
        plt.show()


#What have I learned?
#The main thing is how to set up a unet
#
#I learned the layers of a model object have to be specified in the initialization portion or else
#pytorch does not know about the parameters. You cannot just make the on the fly in "forward". This
#lesson that applies generally as I continue to learn about pytorch neural networks.
#
#I learned you have to use model.eval() and model.train() and with torch.no_grad(): to separate
#the training and evaluation steps. Another general lesson.
#
#I learned about transpose convolution - an important new layer in a NN.
#
#I learned about BCEWithLogitsLoss function. As part of that, I learned the pos_weight
#argument for dealing with unbalanced classes and the fact that it can help or hurt
#depending on your goals for your segmentation problem. If this value is too high, the
#machine just classifies a LOT as polyp to make sure it does not miss anything. Not so
#useful.
#
#I have not done it, but one can also implement, straightforwardly, a probably better
#accuracy measure which is intersection over union (which is the Jaccard index) of the
#positive class. That would be straitforward and the torchmetrics module has a Jaccard
#function but it might be easier to just implement your own in a few lines. Anyway this
#would not be the loss function, just an accuracy measure to do at the end of each
#epoch. You would actually probably not want to replace, so much as complement, the
#pixel accuracy you are already measuring.
#
#I learned you can throw code into ChatGPT and ask for feedback, either generally or help
#with a specific issue, and it works quite well. It found some real bugs in my code and
#identified some important misunderstandings those bugs were based on.
#
#A lot of other small things.