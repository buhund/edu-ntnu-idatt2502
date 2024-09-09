import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Input og mål for NOT-operatoren
inputs = torch.tensor([[0.], [1.]])
targets = torch.tensor([[1.], [0.]])

# Modell
class NOTModel(nn.Module):
    def __init__(self):
        super(NOTModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Instansiering
model = NOTModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss, da vi har 2 klasser
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Trene modellen
epochs = 1000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Visualisere resultater
plt.plot(losses)
plt.title('Loss for NOT-model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
