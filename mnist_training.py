import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchsummary import summary

from model import Model

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

ds_train = DataLoader(ds_train, batch_size=2 ** 12, shuffle=True)
ds_test = DataLoader(ds_test, batch_size=2 ** 12, shuffle=True)


# train model
model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_func = nn.CrossEntropyLoss()

print('summary:')
print(summary(model, (1, 28, 28)))

print("training...")
epochs = 60
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
        
        if i % 10 == 0:
            print(f'batch {i} of {len(ds_train)} ---------------')
            print(f'loss=\t{batch_loss}')
            print(f'accuracy=\t{torch.sum(torch.argmax(pred, dim=1) == labels) / len(labels)}')
            
print("finished training.")


# save model
# eh...
torch.save(model, "model.pth")


# make prediction
print('evaluating performance...')
total_correct_preds = 0
total_datapoints = 0
for data in ds_test:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    pred = model(inputs)
    total_correct_preds += torch.sum(torch.argmax(pred, dim=1) == labels)
    total_datapoints += len(labels)
    
print(f'accuracy=\t{total_correct_preds / total_datapoints}')
