import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Input and target tensors representing the truth table for the NOT operator
# Input: 0 -> Output: 1
# Input: 1 -> Output: 0
inputs = torch.tensor([[0.], [1.]])   # Inputs are single-element tensors (0 and 1)
targets = torch.tensor([[1.], [0.]])  # Corresponding targets are their logical NOT (1 and 0)

# Define the neural network model for learning the NOT operator
class NOTModel(nn.Module):
    def __init__(self):
        super(NOTModel, self).__init__()
        # The model consists of a single linear layer
        # This takes a single input and produces a single output
        self.linear = nn.Linear(1, 1)

    # The forward function defines how the data flows through the model
    # Here, we apply the linear transformation followed by a sigmoid activation
    def forward(self, x):
        # The sigmoid function squashes the output to be between 0 and 1
        # This is useful for binary classification tasks
        return torch.sigmoid(self.linear(x))

# Instantiate the model
model = NOTModel()

# Define the loss function
# Binary Cross Entropy Loss (BCELoss) is used for binary classification tasks
# It calculates how far off the predicted output is from the target for each input
criterion = nn.BCELoss()

# Define the optimizer
# We're using Stochastic Gradient Descent (SGD) to update the model's weights
# 'model.parameters()' passes the model's weights to the optimizer
# lr (learning rate) controls the step size during optimization
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the model
epochs = 1000  # The number of iterations over the dataset
losses = []  # List to store the loss at each epoch

# Loop through each epoch
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients from the previous step

    # Forward pass: Compute the model's predictions for the inputs
    outputs = model(inputs)

    # Compute the loss between the model's predictions (outputs) and the actual targets
    loss = criterion(outputs, targets)

    # Backward pass: Compute the gradients of the loss with respect to the model's parameters
    loss.backward()

    # Update the model's parameters (weights) using the optimizer
    optimizer.step()

    # Store the loss value for later visualization
    losses.append(loss.item())

# Plot the loss over time to visualize training
plt.plot(losses)
plt.title('Loss for NOT-model')  # Title of the graph
plt.xlabel('Epochs')             # X-axis label
plt.ylabel('Loss')               # Y-axis label
plt.show()
