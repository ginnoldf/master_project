import torch.nn as nn
import torch.nn.functional as F


class AtmosphereModel(nn.Module):
    def __init__(self):
        super(AtmosphereModel, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, 5)
        self.conv2 = nn.Conv1d(8, 20, 5)
        self.activ = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(20 * 19, 150)
        self.fc2 = nn.Linear(150, 120)
        self.fc3 = nn.Linear(120, 90)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool(x)
        x = x.view(-1, 20 * 19)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class OceanModel(nn.Module):
    def __init__(self):
        super(OceanModel, self).__init__()
        self.activ = nn.ReLU()
        self.fc1 = nn.Linear(6, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
