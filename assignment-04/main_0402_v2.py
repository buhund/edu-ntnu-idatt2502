import torch
import torch.nn as nn
import torch.optim as optim

### MODEL DATA ###
# Constructs a mapping between letters and int values,
# building a dictionary og letters a-z, a=1, b=2,..., z=26
char_to_index = {chr(i + 96): i for i in range(1, 27)}  # a=1, b=2, ..., z=26
# Sett int value 0 to be space
char_to_index[' '] = 0

# List of emojis to use
emoji_list = ['🎩', '🐀', '🐱', '🏠', '🧑‍🦲', '🧢', '👦']

# Words representing each of the emojis
words = ['hat', 'rat', 'cat', 'flat', 'matt', 'cap', 'son']
labels = [0, 1, 2, 3, 4, 5, 6]

# Changes the words to a sequences of numbers, i.e. each word is converted
# into a sequence of numbers. hat --> h=8, a=1, t=20 --> hat = [8, 1, 20]
sequences = [[char_to_index[char] for char in word] for word in words]

# Pads all words with 0 (space) to be of length 4 if not already
# 'hat' --> 'hat ' --> [8, 1, 20] --> [8, 1, 20, 0]
max_len = 4
for seq in sequences:
    while len(seq) < max_len:
        seq.append(0)

# Create the x and y training tensors
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

# The model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # input -> embedding
        lstm_out, (hidden, _) = self.lstm(embedded)  # LSTM output
        output = self.fc(hidden[-1])  # Bruker siste hidden state for many-to-one
        return output

# Hyperparameters
input_dim = 27  # 26 characters + space
embedding_dim = 8 # Number of relevant characters (emojis)
hidden_dim = 64 # Number of hidden units in the LSTM
output_dim = 7  # 7 classes, 1 for each emoji

# Initialiser modellen, loss-funksjonen og optimizer
model = LSTMModel(input_dim, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

### TRAINING MODEL ###
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# Testing the model
test_words = ['rt', 'rats', 'natt', 'ats']
test_sequences = [[char_to_index[char] for char in word] for word in test_words]
for seq in test_sequences:
    while len(seq) < max_len:
        seq.append(0)

X_test = torch.tensor(test_sequences, dtype=torch.long)

# Evaluation mode. No dropout and batch normalization
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)

# Print emoji based on predictions
for i, word in enumerate(test_words):
    predicted_emoji = emoji_list[predicted_classes[i].item()]
    print(f"Ordet '{word}' er klassifisert som: {predicted_emoji}")