import torch
import numpy as np
from torch import nn
from tqdm import tqdm

try:
    from .model import model
except ImportError:
    from model import model

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

DATASET_DIR = rf'C:\Users\krapa\Downloads\Animals_with_Attributes2\JPEGImages'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 50
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(DATASET_DIR, transform=transform)
class_names = [i.replace('+', '_') for i in dataset.classes]

indices = list(range(len(dataset)))
labels = [dataset.targets[i] for i in indices]

train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    indices, labels, test_size=0.1, stratify=labels, random_state=42
)

val_indices, test_indices, _, _ = train_test_split(
    temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_fn(data_loader, model, optimizer, loss_fn):

    model.train()
    total_loss = 0.0

    for images, labels in tqdm(data_loader):

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
    
    return total_loss / len(data_loader)

def eval_fn(data_loader, model, loss_fn):

    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
        
    return total_loss / len(data_loader), correct / len(data_loader.dataset)

if __name__ == "__main__":
    best_valid_loss = np.inf

    for i in range(EPOCHS):

        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        valid_loss, accuracy = eval_fn(val_loader, model, loss_fn)

        if valid_loss < best_valid_loss: 
            torch.save(model.state_dict(), 'best_model.pt')
            print('SAVED-MODEL')
            best_valid_loss = valid_loss
        
        print(f'Epoch : {i+1:02} Train_loss : {train_loss:.3f} Valid_loss: {valid_loss:.3f}, Accuracy: {accuracy:.3f}')