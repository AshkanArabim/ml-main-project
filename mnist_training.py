import torch
# from torchvision.transforms import functional
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchsummary import summary


# housework
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}.')
transform = transforms.Compose([
    # transforms.functional.pil_to_tensor,
    transforms.ToTensor(),
    # functional.normalize([0.5], ),
])


# load datasets
ds_train = FashionMNIST('.', download=True, train=True, transform=transform)
ds_test = FashionMNIST('.', download=True, train=False, transform=transform)

ds_train = DataLoader(ds_train, batch_size=64, shuffle=True)
ds_test = DataLoader(ds_test, batch_size=64, shuffle=True)


# define model
class Model(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(p=dropout),
            # # nn.Linear(4096, 4096),
            # # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
            nn.Linear(256 * 6 * 6, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# train model
model = Model().to(device)
optimizer = optim.AdamW(model.parameters())
loss_func = nn.CrossEntropyLoss()

print('summary:')
print(summary(model, (1, 28, 28)))

print("training...")
epochs = 100
for epoch in range(epochs):
    print(f'epoch {epoch} or {epochs} ------------------------------------')
    for i, data in enumerate(ds_train):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)
        batch_loss = loss_func(pred, labels)
        batch_loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'batch {i} of {len(ds_train)} ---------------')
            print(f'loss=\t{batch_loss}')
            print(f'accuracy=\t{torch.sum(torch.argmax(pred) == labels) / len(labels)}')
            
print("finished training.")


# save model
# eh...

# working on it -Ale

# make prediction
print('evaluating performance...')
total_correct_preds = 0
total_datapoints = 0
for data in ds_test:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    pred = model(inputs)
    total_correct_preds += torch.sum(torch.argmax(pred) == labels)
    total_datapoints += len(labels)
    
print(f'accuracy=\t{total_correct_preds / total_datapoints}')
