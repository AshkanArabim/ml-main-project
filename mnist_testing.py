import torch, sys, os
from torch.nn import functional
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}.')
transform = transforms.Compose([
    transforms.ToTensor(),
])

ds_test = FashionMNIST('.', download=True, train=False, transform=transform)
ds_test = DataLoader(ds_test, batch_size=2 ** 13, shuffle=True)

name = "None"
model = Model().to('cpu')
check = None

print('Welcome to Model Loader Interface v1 - 4361/5361 ML \nBy Ashkan Arabi, C. Alejandra Carreon & James Newson')

while 1:
    inp = input("\nModel loaded: " + name + "\n\nPlease, select an option (number) from the following menu: \n[1] Load model\n[2] Test model \n[3] Exit program \n>>> ")

    # Load model selected
    if inp == "1":
        check = None
        file = None

        while 1:

            check = input('Please, enter the name of the model file (i.e. "Model_1.pth") or "e" to go back \n>>> ')
            cwd = os.getcwd()

            # Go back statement
            if check == "e":
                break

            file = cwd + "/" + check

            if os.path.exists(file):
                break

            else:
                print("\nERROR: File " + check + " not found at " + cwd + " directory.")

        # Go back statement
        if check == "e":
            continue

        try:
            model = torch.load(file, map_location=torch.device('cpu'))

            name = check
            print("Model \"" + name + "\" loaded succesfully")

        except Exception as error:
            print("ERROR: Model \"" + check + "\" was not able to load; " + str(error))
    else:

        # Test model selected
        if inp == "2":

            if name == "None":
                print("Error: No model loaded.")
                continue

            else:

                while 1:

                    inp = input('Please, select which dataset to test the model with: \n[1] FashionMNIST\n[2] Other Dataset (not available for v1)\n[e] Go back\n>>> ')

                    if inp == "1":

                        print('Evaluating performance on FashionMNIST dataset...')

                        total_correct_preds = 0
                        total_datapoints = 0

                        for data in ds_test:
                            inputs, labels = data
                            inputs = inputs.to('cpu')
                            labels = labels.to('cpu')
                            pred = model(inputs)
                            total_correct_preds += torch.sum(torch.argmax(pred, dim=1) == labels)
                            total_datapoints += len(labels)

                        print(f'---------------\nYielded accuracy: \t{total_correct_preds / total_datapoints} \n---------------')

                    else:

                        if inp == "e":
                            break
                        else:
                            print("\nUnrecognized input.")

        else:

            # Exit selected
            if inp == "3":
                print("Terminating program...")
                sys.exit(0)

            # Any other input
            else:

                print("\nUnrecognized input.")