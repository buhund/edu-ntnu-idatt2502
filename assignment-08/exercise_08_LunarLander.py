# Basic LunarLander env
# Using Gymnasium and Q-Learning

from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

print("Running the LunarLander Agent learning environment: ")

class LunarLanderAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def discretize_state(self, obs, bins=10):
        """Discretizes the continuous state space into bins."""
        state_bins = [np.linspace(low, high, bins) for low, high in zip(self.env.observation_space.low, self.env.observation_space.high)]
        discretized_obs = tuple(np.digitize(ob, bins) for ob, bins in zip(obs, state_bins))
        return discretized_obs

    def get_action(self, obs: np.ndarray) -> int:
        """Chooses action based on epsilon-greedy policy."""
        obs = self.discretize_state(obs)
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: np.ndarray,
    ):
        """Updates Q-values using the Bellman equation."""
        obs = self.discretize_state(obs)  # Discretize current state
        next_obs = self.discretize_state(next_obs)  # Discretize next state

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Decays epsilon to reduce exploration over time."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1

# Initialize environment and agent
#env = gym.make("LunarLander-v3", render_mode="human")  # With fancy graphics. Slower.
env = gym.make("LunarLander-v3")  # Without graphics. Quicker.
env = gym.wrappers.RecordEpisodeStatistics(env)

agent = LunarLanderAgent(
    env=env,  # Pass the environment to the agent
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()  # Reset environment
    done = False

    # Play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # Check if the episode is done
        done = terminated or truncated
        obs = next_obs

    # Decay epsilon after each episode
    agent.decay_epsilon()
