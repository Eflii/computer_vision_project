import torch
import torch.nn as nn

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

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.relu(self.fc1(x))
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
