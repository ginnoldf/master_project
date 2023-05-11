import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, 4)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(4, 8, 4)
        self.fc1 = nn.Linear(8 * 27, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 27)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
