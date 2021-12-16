from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
import numpy as np
from torch.autograd import Variable
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import matplotlib.pyplot as plt


epochs = 100
batch_size = 10
lr = 0.0001
gamma = 0.7
seed = 42
device = "cuda"


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=15):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None


# train and test data directory
data_dir = "/scratch/sg6606/data"
TRANSFORM_IMG = {
    "train": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(contrast = (5)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
}
# load the train and test data
dataset = ImageFolder(data_dir, transform=TRANSFORM_IMG["train"])
print(len(dataset))
train_size = int(6000)
val_size = int(500)
test_size = int(478)

train_data, val_data, test_data = random_split(
    dataset, [train_size, val_size, test_size]
)
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Test Data : {len(test_data)}")

train_dl = DataLoader(
    train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_dl = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_data, batch_size * 2, num_workers=4, pin_memory=True)

model = ViTForImageClassification()    
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()
model.to(device)

train_loss_array = []
train_acc_array = []
val_loss_array = []
val_acc_array = []

for epoch in range(epochs): 
    
    epoch_accuracy=0
    epoch_loss = 0
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    
    for step, (x, y) in enumerate(train_dl):
        x = np.split(np.squeeze(np.array(x)), batch_size)
        for index, array in enumerate(x):
            x[index] = np.squeeze(array)
        x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
        x, y  = x.to(device), y.to(device)
        b_x = Variable(x)   # batch x (image)
        b_y = Variable(y)   # batch y (target)
        output, loss = model(b_x, None)
        if loss is None: 
            loss = loss_func(output, b_y)   
            optimizer.zero_grad()           
            loss.backward()                 
            optimizer.step()
        acc = (output.argmax(dim=1) == y).float().mean()
        epoch_accuracy += acc / len(train_dl)
        epoch_loss += loss / len(train_dl)
    
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")

        
    with torch.no_grad():
        for val_x, val_y in val_dl:
            val_x = np.split(np.squeeze(np.array(val_x)), batch_size)
            for index, array in enumerate(val_x):
                val_x[index] = np.squeeze(array)
            val_x = torch.tensor(np.stack(feature_extractor(val_x)['pixel_values'], axis=0))
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_output, loss = model(val_x, val_y)
            acc = (val_output.argmax(dim=1) == val_y).float().mean()
            epoch_val_accuracy += acc / len(val_dl)
            epoch_val_loss += loss / len(val_dl)
    
    train_loss_array.append(epoch_loss)
    train_acc_array.append(epoch_accuracy)
    val_loss_array.append(epoch_val_loss)
    val_acc_array.append(epoch_val_accuracy)
    
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")


plt.plot(range(1, epochs+1), train_acc_array, label='Training accuracy')
plt.plot(range(1, epochs+1), val_acc_array, label='Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(range(1, epochs+1), train_loss_array, label='Training loss')
plt.plot(range(1, epochs+1), val_loss_array, label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
