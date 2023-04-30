#!/usr/bin/env python
# coding: utf-8

# In[115]:


#! /usr/bin/python3
import numpy as np
from glob import glob
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


# In[116]:


# get_ipython().run_line_magic('cd', "'/workspace'")


# In[117]:


images = glob('{}/*/*_image.jpg'.format('trainval'))
def help(s):
  return s[-51:]
images = sorted(images, key=help)
len(images)




# In[123]:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageDataloader(Dataset):
    """
    Dataloader for Inference.
    """
    def __init__(self, image_paths, target_size=256):

        self.img_paths = image_paths
        self.target_size = target_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.ToPILImage(),transforms.Resize((target_size,target_size)),transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        """
        __getitem__ for inference
        :param idx: Index of the image
        :return: img_np is a numpy RGB-image of shape H x W x C with pixel values in range 0-255.
        And img_tor is a torch tensor, RGB, C x H x W in shape and normalized.
        """
        if len(self.img_paths[idx]) == 2:
            img = cv2.imread(self.img_paths[idx][0])
            label = self.img_paths[idx][1]
        else:
            # test
            img = cv2.imread(self.img_paths[idx])
            label = -1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pad images to target size
        img_tor = self.transform(img).to(device)
        # img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        # img_tor = self.normalize(img_tor)

        # one_hot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=3)
        return img_tor, label

    def __len__(self):
      return len(self.img_paths)


# In[124]:

import pickle
from torch.utils.data import random_split

batch_size = 128
val_size = 1000
train_size = len(images) - val_size 

images = pickle.load(open('tups.pkl', 'rb'))

train_data,val_data = random_split(images,[train_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

train_data = ImageDataloader(train_data)
val_data = ImageDataloader(val_data)

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 0)
# train_dl.train_data.to(torch.device("cuda:0"))
# val_dl.train_data.to(torch.device("cuda:0"))


# In[125]:


import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        print('epoch', epoch)
        
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        print(images.device)
        out = self(images.to(device))                  # Generate predictions
        print(out.device)
        loss = F.cross_entropy(out.to(device), labels.to(device)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out.to(device), labels.to(device))   # Calculate loss
        acc = accuracy(out.to(device), labels.to(device))           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach().cpu()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        torch.save(self.state_dict(), f'{epoch}.ckpt')
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[126]:


class CarClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256 ,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(65536,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,3)
        )
    
    def forward(self, xb):
        return self.network(xb)

model = CarClassification()
model.to(device)


# In[127]:


print(model)


# In[128]:


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001
#fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


