#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from collections import OrderedDict
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # Loading Dataset and Applying transforms
data_dir = "/scratch/nr2387/CVSample/sample/data"
TRANSFORM_IMG = {
                        'train':
                        transforms.Compose([
                            transforms.Resize((224, 224)),
                             transforms.ColorJitter(contrast=(5)),
                            transforms.ToTensor()
                        ])
                    }
#load the train and test data
dataset = ImageFolder(data_dir,transform=TRANSFORM_IMG['train'])
batch_size = 16
print(len(dataset))
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + val_size)

train_data, val_data, test_data = random_split(dataset,[train_size,test_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Validation Data : {len(test_data)}")

train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 1, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 1, pin_memory = True)
test_dl = DataLoader(test_data, batch_size*2, num_workers = 1, pin_memory = True)



model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 15)
model.to(device)


# # Train Model 
optimizer = optim.Adam(model.parameters(),
                       lr = 0.0001)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor = 0.1,
                                                 patience = 4)
epochs = 20
valid_loss_min = np.Inf
weighted_loss=nn.CrossEntropyLoss()

for i in range(epochs):

    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0 

    model.train()
    for images,labels in tqdm(train_dl):
        images = images.to(device)
        labels = labels.to(device)

        ps = model(images)
        loss = weighted_loss(ps,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_dl)

    model.eval()
    with torch.no_grad():
        for images,labels in tqdm(val_dl):
            images = images.to(device)
            labels = labels.to(device)

            ps = model(images)
            loss = weighted_loss(ps,labels)
            valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(val_dl)

    schedular.step(avg_valid_loss)

    if avg_valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(valid_loss_min,avg_valid_loss))
        torch.save({
            'epoch' : i,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'valid_loss_min' : avg_valid_loss
        },'Pneumonia_model.pt')

        valid_loss_min = avg_valid_loss

    print("Epoch : {} Train Loss : {:.6f} ".format(i+1,avg_train_loss))
    print("Epoch : {} Valid Loss : {:.6f} ".format(i+1,avg_valid_loss))


# # Class wise accuracy

def class_accuracy(dataloader, model):

    per_class_accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for images,labels in dataloader:
            ps = model(images.to(device))
            labels = labels.to(device)
            ps = ps.argmax(dim=1).float()
            x1 = ps
            x2 = labels
            per_class_accuracy += int((x1 == x2).sum())

        per_class_accuracy = (per_class_accuracy / len(dataloader.dataset))*100.0

    return per_class_accuracy     


print("Train Dataset Accuracy Report")
acc_list = class_accuracy(train_dl, model)
print(acc_list)

print("Test Dataset Accuracy Report")
acc_list = class_accuracy(test_dl, model)
print(acc_list)

print("Valid Dataset Accuracy Report")
acc_list = class_accuracy(val_dl, model)
print(acc_list)

