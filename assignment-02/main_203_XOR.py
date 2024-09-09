import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Input and target tensors representing the truth table for the XOR operator
# XOR Truth Table:
# Input: (0, 0) -> Output: 0
# Input: (0, 1) -> Output: 1
# Input: (1, 0) -> Output: 1
# Input: (1, 1) -> Output: 0
inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 4 input samples (2D inputs)
targets = torch.tensor([[0.], [1.], [1.], [0.]])  # XOR output

# Define a neural network model to learn the XOR operation with one hidden layer
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Hidden layer: Takes 2 inputs and maps them to 2 outputs
        self.hidden = nn.Linear(2, 2)
        # Output layer: Takes 2 inputs from the hidden layer and outputs 1 value (binary output)
        self.output = nn.Linear(2, 1)

    # Define the forward pass of the network
    def forward(self, x):
        # Apply sigmoid activation after the hidden layer to introduce non-linearity
        x = torch.sigmoid(self.hidden(x))
        # Apply sigmoid activation to the output as well for binary classification
        return torch.sigmoid(self.output(x))

# Instantiate the XOR model
model = XORModel()

# Initialize the weights of the hidden and output layers with random values
# Random initialization helps with breaking symmetry during training
model.hidden.weight.data = torch.Tensor(np.random.uniform(-1, 1, (2, 2)))
model.output.weight.data = torch.Tensor(np.random.uniform(-1, 1, (1, 2)))

# Define the loss function
# BCELoss is used here because we have binary output values
criterion = nn.BCELoss()

# Define the optimizer
# Using Stochastic Gradient Descent (SGD) to update the model's weights
# Learning rate is set to 0.1 to control the step size during optimization
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the model
epochs = 10000  # Number of iterations over the dataset
losses = []  # List to store loss values at each epoch

# Loop through each epoch to train the model
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients from the previous step

    # Forward pass: Compute the model's predictions for the inputs
    outputs = model(inputs)

    # Compute the loss between the predicted outputs and the actual targets
    loss = criterion(outputs, targets)

    # Backward pass: Compute the gradients of the loss with respect to model's parameters
    loss.backward()

    # Update the model's parameters (weights) using the optimizer
    optimizer.step()

    # Store the loss value for this epoch to visualize later
    losses.append(loss.item())

# Plot the loss over time to visualize how the model's performance improved
plt.plot(losses)
plt.title('Loss for XOR-model')  # Title of the plot
plt.xlabel('Epochs')             # Label for the x-axis
plt.ylabel('Loss')               # Label for the y-axis
plt.show()
