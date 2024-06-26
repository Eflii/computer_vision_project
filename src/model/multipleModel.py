import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from data.data_loader import get_data_loaders

# Récupérer les DataLoader
train_loader, test_loader, valid_loader, class_names = get_data_loaders(
    train_dir="../../Dataset/archive/train",
    test_dir="../../Dataset/archive/test",
    valid_dir="../../Dataset/archive/valid"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define models
class CardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CardClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.elu = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.LeakyReLU(self.fc1(x))
        x = self.fc2(x)
        return x

models_dict = {
    "custom_model": CardClassifier(num_classes=53),
    "resnet18": models.resnet18(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
    "densenet121": models.densenet121(pretrained=True),
    "mobilenet_v2": models.mobilenet_v2(pretrained=True)
}


for model_name, model in models_dict.items():
    if model_name != "custom_model":
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, 53)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_features, 53)
            else:
                model.classifier = nn.Linear(model.classifier.in_features, 53)
        models_dict[model_name] = model.to(device)

# Training parameters
criterion = nn.CrossEntropyLoss()
num_epochs = 10
validation_loss_dict = {model_name: [] for model_name in models_dict.keys()}
f1_score_dict = {model_name: [] for model_name in models_dict.keys()}
numberOfData = int(len(train_loader) * 0.1)
numberOfDataValidation = int(len(valid_loader) * 1)

# Training and validation loop
for model_name, model in models_dict.items():
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Training {model_name}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            if i < numberOfData:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        valid_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        validation_loss_dict[model_name].append(epoch_valid_loss)
        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_score_dict[model_name].append(f1)
        print(f"Validation Loss: {epoch_valid_loss:.4f}, F1 Score: {f1:.4f}")

# Plot validation loss
plt.figure(figsize=(20, 5))
for model_name, val_loss in validation_loss_dict.items():
    plt.plot(range(1, num_epochs + 1), val_loss, label=f"{model_name} Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()

# Plot F1 Score
plt.figure(figsize=(20, 5))
for model_name, f1_scores in f1_score_dict.items():
    plt.plot(range(1, num_epochs + 1), f1_scores, label=f"{model_name} F1 Score")
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.show()
