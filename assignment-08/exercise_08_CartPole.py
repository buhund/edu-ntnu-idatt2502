# Basic CartPole env
# Using Gymnasium and Q-Learning

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim

print("Running the CartPole Agent learning environment: ")

# Define the neural network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # First fully connected layer, taking the state as input and producing 64 hidden units
        self.fc1 = nn.Linear(state_size, 64)
        # Second fully connected layer with 64 hidden units
        self.fc2 = nn.Linear(64, 64)
        # Output layer, providing the Q-values for each action
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        # Apply ReLU activation after the first and second layers
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Output Q-values for each action
        return self.fc3(x)

# Hyperparameters
gamma = 0.99  # Discount factor: controls the importance of future rewards
epsilon = 1.0  # Exploration rate: starts with full exploration
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon, reducing exploration over time
learning_rate = 0.001  # Learning rate for the neural network
episodes = 1000  # Number of episodes to train the agent

# Set up the environment (CartPole-v1)
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # The number of state variables (4 for CartPole)
action_size = env.action_space.n  # The number of possible actions (2 for CartPole)

# Instantiate the Q-network and the optimizer
q_network = QNetwork(state_size, action_size)  # Initialize the Q-learning network
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)  # Adam optimizer for updating weights
loss_fn = nn.MSELoss()  # Mean squared error loss to measure the difference between predicted and target Q-values

# Helper function to choose an action using epsilon-greedy policy
def choose_action(state):
    # If a random number is less than epsilon, choose a random action (exploration)
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        # Otherwise, choose the action with the highest Q-value (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to a tensor
            q_values = q_network(state_tensor)  # Get Q-values for all actions
            return torch.argmax(q_values).item()  # Return the action with the highest Q-value

# Training loop for the Q-learning agent
for episode in range(episodes):
    state, _ = env.reset()  # Reset the environment and get the initial state
    total_reward = 0  # Track the total reward for this episode
    done = False  # Keep track of whether the episode has ended

    # Continue the episode until the agent reaches a terminal state
    while not done:
        action = choose_action(state)  # Choose an action based on the current state
        next_state, reward, done, _, _ = env.step(action)  # Take the action and observe the next state and reward
        total_reward += reward  # Accumulate the total reward for this episode

        # Calculate the target Q-value
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Convert next state to tensor
            # Target Q-value is the observed reward plus the discounted future reward, unless the episode is done
            target_q_value = reward + (gamma * torch.max(q_network(next_state_tensor)).item() * (1 - int(done)))

        # Perform Q-network update
        optimizer.zero_grad()  # Clear gradients
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert current state to tensor
        predicted_q_value = q_network(state_tensor)[0, action]  # Get predicted Q-value for the taken action
        loss = loss_fn(predicted_q_value, torch.tensor(target_q_value))  # Calculate the loss between predicted and target Q-values
        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update network weights

        # Move to the next state
        state = next_state

    # Decay epsilon to reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

env.close()  # Close the environment after training is done
