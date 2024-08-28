import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from CSV
data = pd.read_csv('day_head_circumference.csv')

# Prepare the training data
x_train = torch.tensor(data['# day'].values, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data['head circumference'].values, dtype=torch.float32).reshape(-1, 1)

class NonLinearModel:
    def __init__(self):
        # Initialize model parameters
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor using the sigmoid function
    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    # Uses Mean Squared Error as loss function
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

# Initialize the model
model = NonLinearModel()

# Use the Adam optimizer
optimizer = torch.optim.Adam([model.W, model.b], 0.01)

# Train the model
for epoch in range(1000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Update model parameters
    optimizer.zero_grad()  # Clear gradients for next step

# Print model parameters and loss
print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Plot the results
plt.plot(x_train, y_train, 'o', label='Observations')
plt.xlabel('Age (days)')
plt.ylabel('Head Circumference')

# Plot the model prediction
x = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='Model: $f(x) = 20 \cdot \sigma(xW+b) + 31$', color='orange')
plt.legend()
plt.show()

