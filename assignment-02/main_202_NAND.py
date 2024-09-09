import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Input and target tensors representing the truth table for the NAND operator
# Input combinations for two binary inputs and the corresponding NAND outputs
# Input: (0, 0) -> Output: 1
# Input: (0, 1) -> Output: 1
# Input: (1, 0) -> Output: 1
# Input: (1, 1) -> Output: 0
inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 2D input (4 samples, 2 inputs per sample)
targets = torch.tensor([[1.], [1.], [1.], [0.]])  # 1D target output (NAND truth table)

# Define the neural network model to learn the NAND operation
class NANDModel(nn.Module):
    def __init__(self):
        super(NANDModel, self).__init__()
        # The model consists of a single linear layer
        # This layer takes 2 inputs and produces 1 output
        self.linear = nn.Linear(2, 1)

    # The forward function defines how the data flows through the model
    # Here, we apply the linear transformation followed by a sigmoid activation
    def forward(self, x):
        # The sigmoid function squashes the output to a range between 0 and 1
        # This is useful for binary classification tasks like NAND
        return torch.sigmoid(self.linear(x))

# Instantiate the model
model = NANDModel()

# Define the loss function
# Binary Cross Entropy Loss (BCELoss) is used for binary classification
# It compares the predicted output with the target and measures the error
criterion = nn.BCELoss()

# Define the optimizer
# Stochastic Gradient Descent (SGD) is used to update the model's weights
# 'model.parameters()' passes the model's weights to the optimizer
# lr (learning rate) controls the step size during optimization
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the model
epochs = 1000  # Number of iterations over the dataset
losses = []  # List to store the loss at each epoch

# Loop through each epoch
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients to prevent accumulation

    # Forward pass: Compute the model's predictions for the inputs
    outputs = model(inputs)

    # Compute the loss between the model's predictions and the actual targets
    loss = criterion(outputs, targets)

    # Backward pass: Compute the gradients of the loss with respect to the model's parameters
    loss.backward()

    # Update the model's parameters (weights) using the optimizer
    optimizer.step()

    # Store the loss value for later visualization
    losses.append(loss.item())

# Plot the loss over time to visualize training progress
plt.plot(losses)
plt.title('Loss for NAND-model')  # Title of the graph
plt.xlabel('Epochs')              # X-axis label
plt.ylabel('Loss')                # Y-axis label
plt.show()
