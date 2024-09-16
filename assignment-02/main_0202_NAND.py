import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting tools

# Input and target tensors representing the truth table for the NAND operator
inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 2D input (4 samples, 2 inputs per sample)
targets = torch.tensor([[1.], [1.], [1.], [0.]])  # 1D target output (NAND truth table)

# Define the neural network model to learn the NAND operation
class NANDModel(nn.Module):
    def __init__(self):
        super(NANDModel, self).__init__()
        # The model consists of a single linear layer
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        # Apply linear transformation and sigmoid activation
        return torch.sigmoid(self.linear(x))

# Instantiate the model
model = NANDModel()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) # SGD vs Adam?

# Training the model
epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Visualize the loss
plt.figure()
plt.plot(losses)
plt.title('Loss for NAND-model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# 3D Plot of the trained function
# Create a mesh grid of input points for the 3D plot, Should be 01 01 matrix config according to lecture.
x1 = torch.linspace(0, 1, 50)  # Values from 0 to 1 for the first input
x2 = torch.linspace(0, 1, 50)  # Values from 0 to 1 for the second input
X1, X2 = torch.meshgrid(x1, x2)

# Combine x1 and x2 into a single tensor of shape (N, 2)
input_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)

# Predict the outputs using the trained model
with torch.no_grad():
    Z = model(input_grid).reshape(50, 50)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
ax.plot_surface(X1.numpy(), X2.numpy(), Z.numpy())

# Add labels and a title
ax.set_title('3D plot of NAND Model Output')
ax.set_xlabel('Input x1')
ax.set_ylabel('Input x2')
ax.set_zlabel('NAND Output')

plt.show()
