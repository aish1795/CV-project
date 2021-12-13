from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from vit_pytorch.efficient import ViT
from linformer import Linformer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
#train and test data directory
data_dir = "/scratch/ar6381/CV/project/data/sample/data"
TRANSFORM_IMG = {
                        'train':
                        transforms.Compose([
                            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(),
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ])
                    }
#load the train and test data
dataset = ImageFolder(data_dir,transform=TRANSFORM_IMG['train'])
batch_size = 8
print(len(dataset))
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + val_size)

train_data, val_data, test_data = random_split(dataset,[train_size,test_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Validation Data : {len(test_data)}")

train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)
test_dl = DataLoader(test_data, batch_size*2, num_workers = 4, pin_memory = True)

lr = 3e-5
gamma = 0.7
seed = 42
device='cuda'
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=256,
    patch_size=32,
    num_classes=15,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
epochs=10
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_dl):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_dl)
        epoch_loss += loss / len(train_dl)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_dl:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_dl)
            epoch_val_loss += val_loss / len(val_dl)

    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

with torch.no_grad():
	epoch_test_accuracy = 0
	epoch_test_loss = 0
	for data, label in test_dl:
            data = data.to(device)
            label = label.to(device)

            test_output = model(data)
            test_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_dl)
            epoch_test_loss += val_loss / len(test_dl)
	print(f"Test accuracy: {epoch_test_accuracy:.4f}\n")
