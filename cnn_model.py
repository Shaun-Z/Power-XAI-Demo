import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.models as models
from torch_geometric.nn import GATConv
import torch.nn.functional as F


# CNN Model
class VoltageCNN(nn.Module):
    def __init__(self, n_buses):
        super(VoltageCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # Input: (batch, 1, n_buses)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Max pooling (reduces size by half)

        # Placeholder for dynamically computing Flatten dim
        self.flatten_dim = None

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Placeholder; will be adjusted dynamically
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 🔥 Changed to 1 neuron for binary classification

    def forward_features(self, x):
        """Computes CNN feature extraction"""
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, n_buses)
        x = self.forward_features(x)

        # Dynamically compute flatten dimension
        if self.flatten_dim is None:
            self.flatten_dim = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.flatten_dim, 64)  # Now correctly initialized
                
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 🔥 Single neuron output (logit)
        return x 


class VoltageDNN(nn.Module):
    def __init__(self, n_buses):
        super(VoltageDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_buses, 32),  # Fully connected layer (4 -> 32)
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),  # Hidden layer (32 -> 16)
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),  # Output layer (16 -> 1)
        )

    def forward(self, x):
        return self.model(x)


class VoltageGNN(nn.Module):
    def __init__(self, num_buses, hidden_dim=16):
        super(VoltageGNN, self).__init__()
        self.conv1 = GATConv(num_buses, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, 1, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(1)

class VoltageResNet(nn.Module):
    def __init__(self, n_buses):
        super(VoltageResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.resnet(x)
        return x

voltage_data = pd.read_csv('./data/voltage_data.csv').values
voltage_data_outage = pd.read_csv('./data/voltage_data_outage.csv').values

n_buses = voltage_data.shape[1]  # Number of voltage buses
time_steps = voltage_data.shape[0]  # Number of time steps

# 数据集格式转换
class VoltageDataset(Dataset):
    def __init__(self, normal_data, outage_data):
        self.data = []
        self.labels = []

        for i in range(time_steps):  
            self.data.append(normal_data[i].reshape(n_buses))  # 1D CNN 输入
            self.labels.append(0)  # 正常类别

            self.data.append(outage_data[i].reshape(n_buses))  # 1D CNN 输入
            self.labels.append(1)  # Outage 类别

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32).reshape(2*time_steps,1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])