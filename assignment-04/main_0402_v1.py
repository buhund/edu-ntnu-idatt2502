import torch
import torch.nn as nn

# Character encodings (extended for padding)
char_encodings = {
    ' ': [1., 0., 0., 0., 0., 0., 0., 0.],  # Space (padding)
    'h': [0., 1., 0., 0., 0., 0., 0., 0.],
    'a': [0., 0., 1., 0., 0., 0., 0., 0.],
    't': [0., 0., 0., 1., 0., 0., 0., 0.],
    'r': [0., 0., 0., 0., 1., 0., 0., 0.],
    'c': [0., 0., 0., 0., 0., 1., 0., 0.],
    'f': [0., 0., 0., 0., 0., 0., 1., 0.],
    'l': [0., 0., 0., 0., 0., 0., 0., 1.],
    'm': [0., 0., 0., 0., 0., 0., 0.5, 0.],
    's': [0., 0., 0., 0., 0., 0., 0.3, 0.],
    'p': [0., 0., 0., 0., 0., 0., 0.7, 0.],
    'o': [0., 0., 0., 0., 0., 0., 0.2, 0.5],
    'n': [0., 0., 0., 0., 0., 0., 0., 0.9]
}

# Emoji labels
emoji_labels = {
    'hat': 0,   # 🧢
    'rat': 1,   # 🐀
    'cat': 2,   # 🐱
    'flat': 3,  # 🏠
    'matt': 4,  # 🎨
    'cap': 5,   # 🧢
    'son': 6,   # 👶
}

index_to_emoji = ['🧢', '🐀', '🐱', '🏠', '🎨', '👶']

# Pad shorter words with spaces to the max length of 4
max_length = 4
words = ['hat', 'rat', 'cat', 'flat', 'matt', 'cap', 'son']
x_train = [[char_encodings[c] for c in word.ljust(max_length)] for word in words]
y_train = [emoji_labels[word.rstrip()] for word in words]

x_train = torch.tensor(x_train, dtype=torch.float32)  # Shape: (batch_size, sequence_length, encoding_size)
y_train = torch.tensor(y_train, dtype=torch.long)  # Shape: (batch_size)

# Many-to-one LSTM model
class ManyToOneLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManyToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Only use the final output (many-to-one)
        _, (hidden_state, _) = self.lstm(x)
        return self.dense(hidden_state[-1])

# Model setup
model = ManyToOneLSTM(input_size=8, hidden_size=128, output_size=len(emoji_labels))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Test the model
test_words = ['rt', 'rats']
x_test = [[char_encodings[c] for c in word.ljust(max_length)] for word in test_words]
x_test = torch.tensor(x_test, dtype=torch.float32)

# Prediction
with torch.no_grad():
    predictions = model(x_test)
    predicted_emojis = [index_to_emoji[pred.argmax()] for pred in predictions]
    print(f'Predicted emojis: {predicted_emojis}')
