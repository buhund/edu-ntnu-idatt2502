# Basic BlackJack Agent from tutorial
# https://gymnasium.farama.org/introduction/train_agent/
# Using Gymnasium and Q-Learning

from collections import defaultdict, deque
import gymnasium as gym
import numpy as np
from tqdm import tqdm

# Track performance over the last N episodes
reward_window_size = 100  # Moving window size for average reward
reward_history = deque(maxlen=reward_window_size)  # Stores rewards for the last N episodes


print("Running the BlackJack Agent learning environment: ")

# Defining the BlackJack Agent class.
class BlackjackAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        # Q-Values: Store the value of each action in each state. Initialized to zero.
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate # Learning rate for Q-value updates.
        self.discount_factor = discount_factor # Discount for future rewards.


        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

# Initialize environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env)

agent = BlackjackAgent(
    env=env,  # Pass the environment to the agent
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    total_reward = 0 # Track the total reward for this episode.

# play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

        # Accumulate the reward for this episode
        total_reward += reward


    # Store the total reward for this episode
    reward_history.append(total_reward)

    agent.decay_epsilon()

    # Every 100 episodes, print the average reward over the last 100 episodes
    if episode % 1000 == 0 and episode > 0:
        average_reward = np.mean(reward_history)
        print(f"Episode {episode}, Average Reward (last {reward_window_size} episodes): {average_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
