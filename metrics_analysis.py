import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns

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

import matplotlib.pyplot as plt

# Generate confusion matrix
confusion_mat = metrics.confusion_matrix(all_ys, all_y_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")

# Add labels, title, and axis ticks
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.xticks(np.arange(10), [x for x in range (10)], rotation=45)
plt.yticks(np.arange(10), [x for x in range (10)], rotation=0)

# Save the confusion matrix
plt.savefig("./confusion_matrix.png")
plt.close()
