# Assignment 03, exercise a

import torch
import torch.nn as nn
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # Reshape data for Conv2D input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # Reshape data for Conv2D input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

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
        # First model layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 1st Conv Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 1st Pooling Layer

        # Second model layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 2nd Conv Layer
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 2nd Pooling Layer

        # Fully connected layer, adjusted for 64*7*7 input size
        self.dense = nn.Linear(64 * 7 * 7, 128)  # Correct input size for FC layer
        self.out = nn.Linear(128, 10)  # Output layer, mapping to 10 classes

    def logits(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # 1st Conv -> ReLU -> Pool
        x = self.pool2(torch.relu(self.conv2(x)))  # 2nd Conv -> ReLU -> Pool


        x = x.reshape(-1, 64 * 7 * 7)  # Flatten the tensor for the dense layer
        x = torch.relu(self.dense(x))  # Dense layer with ReLU
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
