import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv1d(2, 3, 2)
        self.activ = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(3 * 89, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 90, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(-1, 3 * 89)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CNN_DNN1(nn.Module):
    def __init__(self):
        super(CNN_DNN1, self).__init__()
        self.conv1 = nn.Conv1d(2, 6, 8)
        self.activ = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(6 * 83, 150)
        self.fc2 = nn.Linear(150, 120)
        self.fc3 = nn.Linear(120, 90)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(-1, 6 * 83)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CNN_DNN2(nn.Module):
    def __init__(self):
        super(CNN_DNN2, self).__init__()
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


class DNN1(nn.Module):
    def __init__(self):
        super(DNN1, self).__init__()
        self.fc1 = nn.Linear(180, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 90)

    def forward(self, x):
        x = x.view(-1, 2 * 90)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
