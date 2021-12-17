

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


# In[19]:


epochs = 20
batch_size = 20
lr = 0.0001
gamma = 0.7
seed = 42
device = 'cuda'


# In[13]:


from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=15):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('facebook/dino-vitb8')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None


# In[23]:


data_dir = "/sample/data"
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
batch_size = 10
print(len(dataset))
train_size = 6000
val_size = 500
test_size = len(dataset) - (train_size + val_size)

train_data, val_data, test_data = random_split(dataset,[train_size,val_size,test_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Test Data : {len(test_data)}")

train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 1, pin_memory = True)
val_dl = DataLoader(val_data, batch_size, num_workers = 1, pin_memory = True)
test_dl = DataLoader(test_data, batch_size*2, num_workers = 1, pin_memory = True)


# In[16]:


model = ViTForImageClassification()
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()
model.to(device)


# In[ ]:


train_loss_array = []
train_acc_array = []
val_loss_array = []
val_acc_array = []
valid_loss_min = np.Inf
for epoch in range(epochs):

    epoch_accuracy=0
    epoch_loss = 0
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    model.train()

    for step, (x, y) in enumerate(train_dl):
#         print('Train start')
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
#         print('End')

    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
    f_path = 'checkpoint.pt'
    torch.save(checkpoint, f_path)
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_dl:
#             print(val_x.shape)
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
    if epoch_val_loss < valid_loss_min:
        print("Saving Model")
        torch.save(model.state_dict(), 'best_model_DINO.pth')
        valid_loss_min = epoch_val_loss

    train_loss_array.append(epoch_loss)
    train_acc_array.append(epoch_accuracy)
    val_loss_array.append(epoch_val_loss)
    val_acc_array.append(epoch_val_accuracy)

    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")


with torch.no_grad():
	epoch_test_accuracy = 0
	epoch_test_loss = 0
	for data, label in test_dl:
            data = data.to(device)
            label = label.to(device)
            test_output = model(data)
            test_loss = criterion(test_output, label)
            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_dl)
            epoch_test_loss += val_loss / len(test_dl)
	print(f"Test accuracy: {epoch_test_accuracy:.4f}\n")

print(train_loss_array)
print(train_acc_array)
print(val_loss_array)
print(val_acc_array)



# In[ ]:
