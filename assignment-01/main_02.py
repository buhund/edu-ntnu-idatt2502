import torch
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Observed/training data from CSV file
data = pd.read_csv('day_length_weight.csv')
x_train = torch.tensor(data[['length', 'weight']].values, dtype=torch.float32)
y_train = torch.tensor(data['# day'].values, dtype=torch.float32).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.zeros((2, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numerical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], 0.01) # Changed optimizer from SGD to Adam
for epoch in range(1000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train).item()))

# Plot the results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['length'], data['weight'], data['# day'], label='Observations')

# Create a mesh grid for plotting the plane
x1, x2 = torch.meshgrid(torch.linspace(data['length'].min(), data['length'].max(), 10),
                        torch.linspace(data['weight'].min(), data['weight'].max(), 10))
y = model.f(torch.cat((x1.reshape(-1, 1), x2.reshape(-1, 1)), dim=1)).detach().reshape(10, 10)

ax.plot_surface(x1.numpy(), x2.numpy(), y.numpy(), color='orange', alpha=0.5, label='Model')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Age (days)')
plt.legend()
plt.show()
