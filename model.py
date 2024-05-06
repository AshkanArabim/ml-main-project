import torch
from torch import nn
from torch.nn import functional

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = self.pool(x)
        x = functional.relu(self.conv2(x))
        x = self.pool(x)
        x = functional.relu(self.conv3(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
