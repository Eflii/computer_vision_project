import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Définir les transformations pour prétraiter les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Téléchargement du dataset
test_dataset = datasets.ImageFolder(root="../Dataset/archive/train", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Charger le modèle VGG16 pré-entraîné
vgg16 = models.vgg16(pretrained=True)

# Modifier la couche finale pour qu'elle corresponde au nombre de classes de votre dataset
num_classes = 53
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, num_classes)

# Charger le modèle sur l'appareil (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

# Charger les poids enregistrés si nécessaire
# saved_model_path = "path/to/saved_model.pth"
# vgg16.load_state_dict(torch.load(saved_model_path))

vgg16.eval()

# Faire des prédictions sur l'ensemble de test
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculer la matrice de confusion
cm = confusion_matrix(true_labels, predicted_labels)

# Classes de votre dataset
class_names = test_dataset.classes

# Visualiser la matrice de confusion avec seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Valeurs Prédites')
plt.ylabel('Valeurs Réelles')
plt.title('Confusion_matrix')
plt.show()
