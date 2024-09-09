import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.utils.data

# Path for storing data sets
data_path = './data/model_weights/'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Defining the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.linear1(x))
        return torch.softmax(self.linear2(x), dim=1)

model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Visualization of an example from the data set
example_data, example_targets = next(iter(trainloader))
plt.imshow(example_data[0][0], cmap='gray')
plt.title(f'Number: {example_targets[0]}')
plt.show()

# Stor weight images after otpimization
weights = model.linear1.weight.data
for i in range(10):
    plt.imshow(weights[i].view(28, 28).cpu().numpy(), cmap='gray')
    plt.title(f'Weight for class {i}')
    plt.savefig(f'./data/model_weights/weight_class_{i}.png')
    plt.show()

