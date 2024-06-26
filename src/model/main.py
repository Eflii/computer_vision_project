import os
import random
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from data.data_loader import get_data_loaders
from model import CardClassifier

# Charger les DataLoader
train_loader, test_loader, valid_loader, class_names = get_data_loaders(
    train_dir="../../Dataset/archive/train",
    test_dir="../../Dataset/archive/test",
    valid_dir="../../Dataset/archive/valid"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancier le modèle
model = CardClassifier(num_classes=53).to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 5
percentage_of_data = 1  # Ajustez ce pourcentage si nécessaire
numberOfData = int(len(train_loader) * percentage_of_data)
start_time = time.time()

epoch_tab, loss_function, accuracy, F1_score, f1OverEpoch = [], [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    predictions, true_labels = [], []
    epoch_tab.append(epoch)

    for i, (inputs, labels) in enumerate(train_loader):
        if i < numberOfData:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / (numberOfData * train_loader.batch_size)
    loss_function.append(epoch_loss)
    accuracy.append(correct / total)
    f1 = f1_score(true_labels, predictions, average="macro")
    F1_score.append(f1)

    model.eval()
    true_labels_valid, predicted_labels = [], []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels_valid.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted.cpu().tolist())

    f1 = f1_score(true_labels_valid, predicted_labels, average='macro')
    f1OverEpoch.append(f1)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, F1 Score: {f1:.4f}")

plt.plot(epoch_tab, f1OverEpoch)
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.show()

# Enregistrer le modèle
identifier = str(random.randint(1, 10000))
directory = identifier
parent_dir = "results"
os.makedirs(os.path.join(parent_dir, directory), exist_ok=True)

plt.plot(epoch_tab, loss_function)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss during epoch")
plt.savefig(os.path.join(parent_dir, directory, "loss_over_time.png"), bbox_inches='tight')
plt.show()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Sauvegarder le modèle
filename = os.path.join(parent_dir, directory, "saved_model.pth")
torch.save(model.state_dict(), filename)
