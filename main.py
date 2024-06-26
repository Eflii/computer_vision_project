import os
import random
from PIL import Image
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Définition des transformations pour prétraiter les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Téléchargement du dataset (à adapter en fonction de l'emplacement de vos données)
train_dataset = datasets.ImageFolder(root="Dataset/archive/train", transform=transform)
test_dataset = datasets.ImageFolder(root="Dataset/archive/test", transform=transform)

# Définition du DataLoader pour charger les données en batch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définition de l'architecture du modèle de classification
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
        #print(x.shape)
        x = x.view(-1, 64 * 28 * 28)
        x = self.LeakyReLU(self.fc1(x))
        x = self.fc2(x)

        return x

    def intermediate(self, x):
        intermediate_outputs = []
        x = self.relu(self.conv1(x))
        intermediate_outputs.append(x)
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        intermediate_outputs.append(x)
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        intermediate_outputs.append(x)
        x = self.pool(x)

        return intermediate_outputs


# Instanciation du modèle
model = CardClassifier(num_classes=53)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = model.to(device)

saved_model_path = "training_session/8558/savedModel_8558.pth"
model.load_state_dict(torch.load(saved_model_path, map_location=device))



identifier = str(random.randint(1, 10000))
# Directory
directory = identifier

# Parent Directory path
parent_dir = r"C:\Users\hugoe\Rangement\Cours\Algebra\ComputerVIsion\ComputerVIsion\pythonProject"


# mode
mode = 0o666

# Path
path = os.path.join(parent_dir, directory)

# Create the directory
# 'GeeksForGeeks' in
# '/home / User / Documents'
# with mode 0o666
os.mkdir(path, mode)
print("Directory '% s' created" % directory)

# Évaluation du modèle sur l'ensemble de test

# Save the model
from sklearn.metrics import f1_score

model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

f1 = f1_score(true_labels, predicted_labels, average='macro')  # or 'micro', 'weighted', depending on your preference
print(f"F1 score on test set: {f1:.2f}")

# Fonction pour afficher les feature maps après chaque couche de convolution
def visualize_feature_maps(image_path, model):
    # Charger l'image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    # Activer le mode évaluation
    model.eval()

    # Obtenir les activations de chaque couche de convolution
    activations = model.intermediate(image_tensor)

    # Afficher les feature maps
    for i, activation in enumerate(activations):
        plt.figure(figsize=(10, 5))
        plt.title(f"Layer {i+1} - {activation.shape}")
        num_feature_maps = min(activation.shape[1], 32)
        for j in range(num_feature_maps):
            plt.subplot(4, 8, j+1)
            plt.imshow(activation[0, j].detach().cpu(), cmap='viridis')
            plt.axis('off')
        plt.show()

# Fonction pour afficher les prédictions du modèle sur certaines images
def visualize_predictions(model, dataloader, num_images=5):
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_images:
                break
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print(f"Predicted: {predicted.numpy()}, '\n' Actual: {labels.numpy()}")

# Utilisation des fonctions
image_path = r"C:\Users\hugoe\Rangement\Cours\Algebra\ComputerVIsion\ComputerVIsion\pythonProject\Dataset\archive\train\eight of diamonds\005.jpg"
visualize_feature_maps(image_path, model)

# Afficher les prédictions sur certaines images
visualize_predictions(model, test_loader, num_images=4)