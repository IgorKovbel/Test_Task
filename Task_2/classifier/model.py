from torch import nn
from torchvision import models

# Download ResNet50 with partial freezing of weights
model = models.resnet50(weights="IMAGENET1K_V2")
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layers
for param in model.layer4.parameters():
    param.requires_grad = True

# New layers for classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.45),
    nn.Linear(256, 50)
)