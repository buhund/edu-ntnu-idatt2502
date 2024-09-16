import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sigmoid function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# Model for NOT-operator
class SigmoidModel:
    def __init__(self):
        # Initializing weight and bias
        self.W = np.array([[-10.0]])  # Weight for separation
        self.b = np.array([[5.0]])    # Bias to promote desired output.

    # Predictor
    def f(self, x):
        return sigmoid(x @ self.W + self.b)

    # Use Cross Entropy as loss function
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))

model = SigmoidModel()

# Input (x) and output (y) for NOT-operator, 01 10 as per lecture.
x_train = np.array([[0], [1]])
y_train = np.array([[1], [0]])

# 3D plot (I hope)
fig = plt.figure("Logistic regression: the logical NOT operator")

plot1 = fig.add_subplot(111, projection='3d')

# Plot training data
plot1.scatter(x_train[:, 0].squeeze(), x_train[:, 0].squeeze(), y_train[:, 0].squeeze(), color="blue", label="Training data")

plot1.set_xlabel("$x$")
plot1.set_ylabel("$x$")
plot1.set_zlabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

def update_figure():
    x_grid = np.linspace(-0.25, 1.25, 50)
    x1_grid, x2_grid = np.meshgrid(x_grid, x_grid)
    y_grid = model.f(x1_grid.reshape(-1, 1)).reshape(x1_grid.shape)

    plot1.clear()

    plot1.scatter(x_train[:, 0].squeeze(), x_train[:, 0].squeeze(), y_train[:, 0].squeeze(), color="blue", label="Training data")

    plot1.plot_wireframe(x1_grid, x2_grid, y_grid, alpha=0.6)

    # Oppdaterer aksene
    plot1.set_xlabel("$x$")
    plot1.set_ylabel("$x$")
    plot1.set_zlabel("$y$")
    plot1.set_xticks([0, 1])
    plot1.set_yticks([0, 1])
    plot1.set_zticks([0, 1])
    plot1.set_xlim(-0.25, 1.25)
    plot1.set_ylim(-0.25, 1.25)
    plot1.set_zlim(-0.25, 1.25)
    plot1.set_title(f"W = {model.W[0, 0]:.2f}, b = {model.b[0, 0]:.2f}, Loss = {model.loss(x_train, y_train):.2f}")

    plt.pause(0.01)
    fig.canvas.draw()

# Visualiserer modellen etter optimalisering
update_figure()
plt.show()
