import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

from model import Model


# housework
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}.')
transform = transforms.Compose([
    transforms.ToTensor(),
])


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
