import torch
from torch.nn import functional
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch import nn
import sklearn.metrics as metrics


# housework
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}.')
transform = transforms.Compose([
    transforms.ToTensor(),
])


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

model = torch.load("./model.pth")


print('calculating metrics...')

ds_test = FashionMNIST('.', download=True, train=False, transform=transform)
ds_test = DataLoader(ds_test, batch_size=2 ** 13, shuffle=True)

all_ys = []
all_y_preds = []

for data in ds_test:
    inputs, labels = data
    
    all_ys += labels
    all_y_preds += torch.argmax(model(inputs.to(device)), dim=1).to("cpu")
    
report = metrics.classification_report(all_ys, all_y_preds)
print(report)
