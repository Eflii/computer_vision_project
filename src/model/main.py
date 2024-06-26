import os
import random
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from PIL import Image
from torchvision import transforms

# Assumez que vous avez déjà défini vos fonctions et importé vos modules comme indiqué précédemment...

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data.data_loader import get_data_loaders
from model import CardClassifier

# get the data
train_loader, test_loader, valid_loader, class_names = get_data_loaders(
    train_dir="../../Dataset/archive/train",
    test_dir="../../Dataset/archive/test",
    valid_dir="../../Dataset/archive/valid"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the model architecture
model = CardClassifier(num_classes=53).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training part
num_epochs = 2
percentage_of_data = 1
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

# sabe the model
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

# Sauvegarder le modèle entraîné
filename = os.path.join(parent_dir, directory, "saved_model.pth")
torch.save(model.state_dict(), filename)

# evaluate the model
model.eval()
true_labels, predicted_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predicted.cpu().tolist())

f1 = f1_score(true_labels, predicted_labels, average='macro')
print(f"F1 score on test set: {f1:.2f}")


# feature map function
def visualize_feature_maps(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    activations = model.intermediate(image_tensor)
    for i, activation in enumerate(activations):
        num_feature_maps = activation.shape[1]
        num_cols = 8
        num_rows = (num_feature_maps // num_cols) + int(num_feature_maps % num_cols != 0)
        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        plt.suptitle(f"Layer {i + 1} - {activation.shape}")
        for j in range(num_feature_maps):
            plt.subplot(num_rows, num_cols, j + 1)
            # Detach the tensor from the computation graph before converting to numpy
            plt.imshow(activation[0, j].detach().cpu().numpy(), cmap='viridis')
            plt.axis('off')
        plt.show()


image_path = "../../Dataset/archive/train/eight of diamonds/005.jpg"
visualize_feature_maps(image_path, model)


# visualization of some prediction
def visualize_predictions(model, dataloader, class_names, num_images=5):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 15))
    with torch.no_grad():
        dataloader_iter = iter(dataloader)
        while images_so_far < num_images:
            try:
                images, labels = next(dataloader_iter)
                outputs = model(images.to(device))
                _, preds = torch.max(outputs, 1)
                for i in range(images.size(0)):
                    if images_so_far >= num_images:
                        break
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'Predicted: {class_names[preds[i]]}\nActual: {class_names[labels[i]]}')
                    img = images[i].cpu().numpy().transpose((1, 2, 0))
                    img = img * 0.229 + 0.485  # Unnormalize
                    plt.imshow(img)
            except StopIteration:
                break
    plt.tight_layout()
    plt.show()

visualize_predictions(model, test_loader, class_names, num_images=8)

