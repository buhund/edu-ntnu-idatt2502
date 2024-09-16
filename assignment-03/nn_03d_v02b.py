# Assignment 03, exercise b

import torch
import torch.nn as nn
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train_fashion = torchvision.datasets.FashionMNIST('./data/fashion', train=True, download=True)
x_train = mnist_train_fashion.data.reshape(-1, 1, 28, 28).float()  # Reshape data for Conv2D input
y_train = torch.zeros((mnist_train_fashion.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train_fashion.targets.shape[0]), mnist_train_fashion.targets] = 1  # Populate output

mnist_test_fashion = torchvision.datasets.FashionMNIST('./data/fashion', train=False, download=True)
x_test = mnist_test_fashion.data.reshape(-1, 1, 28, 28).float()  # Reshape data for Conv2D input
y_test = torch.zeros((mnist_test_fashion.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test_fashion.targets.shape[0]), mnist_test_fashion.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        # First Convolution and Pooling layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 1st Conv Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 1st Pooling Layer

        # Second Convolution and Pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 2nd Conv Layer
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 2nd Pooling Layer

        # Fully connected layer, fc
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 128)  # Correct input size for FC layer
        self.out = nn.Linear(128, 10) # Output layer

    def logits(self, x):
        # Pass through the convolutional and pooling layers
        x = self.pool1(torch.relu(self.conv1(x)))  # 1st Conv -> ReLU -> Pool
        x = self.pool2(torch.relu(self.conv2(x)))  # 2nd Conv -> ReLU -> Pool

        x = x.reshape(-1, 64 * 7 * 7)  # Flatten the tensor for the fully connected layer

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))  # W3 Dense 1024 neurons with ReLU
        x = torch.relu(self.fc2(x))  # Fully connected layer (128 neurons) with ReLU
        return self.out(x)  # Output layer

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Optimizer: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Training loop
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        optimizer.zero_grad()  # Clear gradients for next step
        loss = model.loss(x_train_batches[batch], y_train_batches[batch])
        loss.backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b

    # print("accuracy = %s" % model.accuracy(x_test, y_test))
    print(f"Epoch {epoch + 1}, accuracy = {model.accuracy(x_test, y_test).item() * 100:.2f}%")


# Model: B
# Epochs: 20
# Accuracy for FashionMNIST: