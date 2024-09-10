import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, art3d

matplotlib.rcParams.update({'font.size': 12, 'figure.figsize': (14, 8)})

W1_init = np.array([[10.0, -10.0], [10.0, -10.0]])
b1_init = np.array([[-5.0, 15.0]])
W2_init = np.array([[10.0], [10.0]])
b2_init = np.array([[-15.0]])

# Also try:
# W1_init = np.array([[7.43929911, 5.68582106], [7.44233704, 5.68641663]])
# b1_init = np.array([[-3.40935969, -8.69532299]])
# W2_init = np.array([[13.01280117], [-13.79168701]])
# b2_init = np.array([[-6.1043458]])


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class SigmoidModel:
    def __init__(self, W1=W1_init.copy(), W2=W2_init.copy(), b1=b1_init.copy(), b2=b2_init.copy()):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    # First layer function
    def f1(self, x):
        return sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


model = SigmoidModel()

# Observed/training input and output
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

fig = plt.figure("Logistic regression: the logical XOR operator")

plot1 = fig.add_subplot(131, projection='3d')

plot1.plot_wireframe(
    np.array([[]]),
    np.array([[]]),
    np.array([[]]),
    color="green",
    label="$\\mathbf{h}=\\mathrm{f1}(\\mathbf{x})=\\sigma(\\mathbf{x W1}+\\mathbf{b1})$",
)
plot1_h1 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))
plot1_h2 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))

plot1_info = fig.text(0.15, 0.9, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$h_1,h_2$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

plot2 = fig.add_subplot(132, projection='3d')

plot2_f2 = plot2.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$\\mathrm{f2}(\\mathbf{h})=\\sigma(\\mathbf{h W2}+b2)$")

plot2_info = fig.text(0.45, 0.9, "")

plot2.set_xlabel("$h_1$")
plot2.set_ylabel("$h_2$")
plot2.set_zlabel("$y$")
plot2.legend(loc="upper left")
plot2.set_xticks([0, 1])
plot2.set_yticks([0, 1])
plot2.set_zticks([0, 1])
plot2.set_xlim(-0.25, 1.25)
plot2.set_ylim(-0.25, 1.25)
plot2.set_zlim(-0.25, 1.25)

plot3 = fig.add_subplot(133, projection='3d')

plot3_f = plot3.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$f(\\mathbf{x})=\\mathrm{f2}(\\mathrm{f1}(\\mathbf{x}))$")

plot3.plot(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)}, y^{(i)})$", color="blue")

plot3_info = fig.text(0.15, 0.04, "")

plot3.set_xlabel("$x_1$")
plot3.set_ylabel("$x_2$")
plot3.set_zlabel("$y$")
plot3.legend(loc="upper left")
plot3.set_xticks([0, 1])
plot3.set_yticks([0, 1])
plot3.set_zticks([0, 1])
plot3.set_xlim(-0.25, 1.25)
plot3.set_ylim(-0.25, 1.25)
plot3.set_zlim(-0.25, 1.25)

table = plt.table(
    cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
    colWidths=[0.15] * 3,
    colLabels=["$x_1$", "$x_2$", "$f(\\mathbf{x})$"],
    cellLoc="center",
    bbox=[0.925, -0.2, 0.4, 0.4]
)


def update_figure(event=None):
    if (event is not None):
        if event.key == "W":
            model.W1[0, 0] += 0.2
        elif event.key == "w":
            model.W1[0, 0] -= 0.2
        elif event.key == "E":
            model.W1[0, 1] += 0.2
        elif event.key == "e":
            model.W1[0, 1] -= 0.2
        elif event.key == "R":
            model.W1[1, 0] += 0.2
        elif event.key == "r":
            model.W1[1, 0] -= 0.2
        elif event.key == "T":
            model.W1[1, 1] += 0.2
        elif event.key == "t":
            model.W1[1, 1] -= 0.2

        elif event.key == "Y":
            model.W2[0, 0] += 0.2
        elif event.key == "y":
            model.W2[0, 0] -= 0.2
        elif event.key == "U":
            model.W2[1, 0] += 0.2
        elif event.key == "u":
            model.W2[1, 0] -= 0.2

        elif event.key == "B":
            model.b1[0, 0] += 0.2
        elif event.key == "b":
            model.b1[0, 0] -= 0.2
        elif event.key == "N":
            model.b1[0, 1] += 0.2
        elif event.key == "n":
            model.b1[0, 1] -= 0.2

        elif event.key == "M":
            model.b2[0, 0] += 0.2
        elif event.key == "m":
            model.b2[0, 0] -= 0.2

        elif event.key == "c":
            model.W1 = W1_init.copy()
            model.W2 = W2_init.copy()
            model.b1 = b1_init.copy()
            model.b2 = b2_init.copy()

    global plot1_h1, plot1_h2, plot2_f2, plot3_f
    plot1_h1.remove()
    plot1_h2.remove()
    plot2_f2.remove()
    plot3_f.remove()
    x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
    h1_grid = np.empty([10, 10])
    h2_grid = np.empty([10, 10])
    f2_grid = np.empty([10, 10])
    f_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            h = model.f1([[x1_grid[i, j], x2_grid[i, j]]])
            h1_grid[i, j] = h[0, 0]
            h2_grid[i, j] = h[0, 1]
            f2_grid[i, j] = model.f2([[x1_grid[i, j], x2_grid[i, j]]])
            f_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])

    plot1_h1 = plot1.plot_wireframe(x1_grid, x2_grid, h1_grid, color="lightgreen")
    plot1_h2 = plot1.plot_wireframe(x1_grid, x2_grid, h2_grid, color="darkgreen")

    plot1_info.set_text(
        "$\\mathbf{W1}=\\left[\\genfrac{}{}{0}{}{%.2f}{%.2f}\\/\\genfrac{}{}{0}{}{%.2f}{%.2f}\\right]$\n$\\mathbf{b1}=[{%.2f}, {%.2f}]$" %
        (model.W1[0, 0], model.W1[1, 0], model.W1[0, 1], model.W1[1, 1], model.b1[0, 0], model.b1[0, 1])
    )

    plot2_f2 = plot2.plot_wireframe(x1_grid, x2_grid, f2_grid, color="green")

    plot2_info.set_text("$\\mathbf{W2}=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\nb2$=[{%.2f}]$" % (model.W2[0, 0], model.W2[1, 0], model.b2[0, 0]))

    plot3_f = plot3.plot_wireframe(x1_grid, x2_grid, f_grid, color="green")

    plot3_info.set_text(
        "$loss = -\\frac{1}{N}\\sum_{i=1}^{N}\\left [ y^{(i)} \\log\\/f(\\mathbf{x}^{(i)}) + (1-y^{(i)}) \\log (1-f(\\mathbf{x}^{(i)})) \\right ] = %.2f$" %
        model.loss(x_train, y_train)
    )

    table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f([[0, 0]]))
    table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f([[0, 1]]))
    table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f([[1, 0]]))
    table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f([[1, 1]]))

    plt.pause(0.01)
    fig.canvas.draw()


update_figure()
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()
