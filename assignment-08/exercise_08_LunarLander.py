from collections import defaultdict, deque
import gymnasium as gym
import numpy as np
from tqdm import tqdm

# Track performance over the last N episodes
reward_window_size = 100  # Moving window size for average reward
reward_history = deque(maxlen=reward_window_size)  # Stores rewards for the last N episodes


# Class definition for the LunarLander agent
class LunarLanderAgent:
    def __init__(
            self,
            env: gym.Env,  # The Gymnasium environment
            learning_rate: float,  # Learning rate (alpha) for Q-learning updates
            initial_epsilon: float,  # Initial epsilon for epsilon-greedy exploration
            epsilon_decay: float,  # Rate at which epsilon decays over episodes
            final_epsilon: float,  # Minimum value for epsilon (ensures some exploration)
            discount_factor: float = 0.95,  # Discount factor (gamma) for future rewards
    ):
        """
        Initialize the Lunar Lander agent with an environment, learning rate,
        and parameters for epsilon-greedy exploration and discounting future rewards.

        Args:
            env: The Lunar Lander Gym environment.
            learning_rate: Learning rate (alpha) for Q-learning updates.
            initial_epsilon: The starting value for epsilon in epsilon-greedy exploration.
            epsilon_decay: The rate at which epsilon decreases over episodes.
            final_epsilon: The minimum value that epsilon can decay to.
            discount_factor: Discount factor for future rewards (gamma), typically between 0.9 and 0.99.
        """
        self.env = env
        # Q-values stored in a dictionary where each key is a state (discretized) and
        # each value is an array of action values (size of number of actions in the environment).
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # Hyperparameters for the Q-learning update
        self.lr = learning_rate  # Learning rate for Q-value updates
        self.discount_factor = discount_factor  # Discount factor for future rewards

        # Epsilon-greedy parameters for exploration vs. exploitation
        self.epsilon = initial_epsilon  # Starting epsilon (probability of exploring)
        self.epsilon_decay = epsilon_decay  # Amount by which epsilon decays each episode
        self.final_epsilon = final_epsilon  # Minimum value epsilon can decay to

        # Keep track of training error for each update (can be useful for debugging)
        self.training_error = []

    def discretize_state(self, obs, bins=10):
        """
        Discretize the continuous state space into discrete bins.
        This is important for using Q-learning, since the state space is continuous.
        The state space in Lunar Lander consists of 8 continuous variables.

        Args:
            obs: The observation (state) from the environment (continuous values).
            bins: Number of discrete bins to divide the continuous state space into.

        Returns:
            discretized_obs: A tuple representing the discretized state,
                             where each component of the observation is discretized.
        """
        # Create bins based on the range of each observation component (min/max) for the environment
        state_bins = [np.linspace(low, high, bins) for low, high in zip(self.env.observation_space.low, self.env.observation_space.high)]

        # Discretize each component of the observation using the bins created
        discretized_obs = tuple(np.digitize(ob, bins) for ob, bins in zip(obs, state_bins))
        return discretized_obs

    def get_action(self, obs: np.ndarray) -> int:
        """
        Choose the next action using an epsilon-greedy strategy:
        - With probability epsilon, select a random action to encourage exploration.
        - Otherwise, select the action that has the highest Q-value for the current state.

        Args:
            obs: The current observation (state) from the environment.

        Returns:
            action: The chosen action (either random or greedy).
        """
        # Discretize the continuous observation to match the state space in the Q-table
        obs = self.discretize_state(obs)

        # With probability epsilon, take a random action (explore)
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        # Otherwise, choose the best action based on learned Q-values (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: np.ndarray,  # The current observation (state)
            action: int,  # The action taken
            reward: float,  # The reward received after taking the action
            terminated: bool,  # Whether the episode has ended
            next_obs: np.ndarray,  # The next observation (state) after taking the action
    ):
        """
        Perform a Q-learning update by adjusting the Q-value for the (state, action) pair.

        Args:
            obs: The current observation (state).
            action: The action taken in the current state.
            reward: The reward received after taking the action.
            terminated: A boolean indicating if the episode has ended (terminal state).
            next_obs: The next observation (state) after taking the action.
        """
        # Discretize the current and next states
        obs = self.discretize_state(obs)
        next_obs = self.discretize_state(next_obs)

        # Compute the maximum Q-value for the next state (used in Bellman equation)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # Compute the temporal difference (TD) error: difference between current and future Q-value estimates
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        # Update the Q-value for the current state-action pair using the TD error and learning rate
        self.q_values[obs][action] += self.lr * temporal_difference

        # Keep track of training errors for monitoring purposes (optional)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        Decay epsilon to gradually reduce exploration over time and favor exploitation.
        Epsilon will decay until it reaches a minimum value (final_epsilon).
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Hyperparameters for the Q-learning agent
learning_rate = 0.01  # Learning rate (alpha)
n_episodes = 100_0  # Total number of episodes for training
start_epsilon = 1.0  # Initial epsilon (exploration rate)
epsilon_decay = start_epsilon / (n_episodes / 2)  # Epsilon decay rate (reduce exploration over time)
final_epsilon = 0.1  # Minimum epsilon (ensures some exploration continues)

# Initialize the LunarLander environment (with or without graphics)
# With render_mode="human", the environment shows graphics, which can slow down training.
# To speed up training, omit the render_mode or set it to None.
# env = gym.make("LunarLander-v3", render_mode="human")  # With graphics (slower)
env = gym.make("LunarLander-v3")  # Without graphics (faster)

# Wrapper to keep track of episode statistics (e.g., rewards) during training
env = gym.wrappers.RecordEpisodeStatistics(env)

# Initialize the LunarLanderAgent with the environment and hyperparameters
agent = LunarLanderAgent(
    env=env,  # The environment in which the agent operates
    learning_rate=learning_rate,  # The learning rate for Q-value updates
    initial_epsilon=start_epsilon,  # Starting exploration rate (epsilon)
    epsilon_decay=epsilon_decay,  # Rate at which epsilon decays over episodes
    final_epsilon=final_epsilon,  # Minimum epsilon value for exploration
)

# Main training loop for the Q-learning agent
for episode in tqdm(range(n_episodes)):  # tqdm provides a progress bar for the loop
    obs, info = env.reset()  # Reset the environment to start a new episode
    done = False  # Track whether the episode has finished
    total_reward = 0 # Track the total reward for this episode

    # Play one episode (loop until the episode ends)
    while not done:
        action = agent.get_action(obs)  # Get the action to take based on the current state
        next_obs, reward, terminated, truncated, info = env.step(action)  # Take the action in the environment

        # Update the agent's Q-value based on the experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Accumulate the reward for this episode
        total_reward += reward

        # Check if the episode is done (either terminated or truncated)
        done = terminated or truncated
        obs = next_obs  # Update the observation to the next state

    # Store the total reward for this episode
    reward_history.append(total_reward)

    # After each episode, decay epsilon to reduce exploration over time
    agent.decay_epsilon()

    # Every 100 episodes, print the average reward over the last 100 episodes
    if episode % 100 == 0 and episode > 0:
        average_reward = np.mean(reward_history)
        print(f"Episode {episode}, Average Reward (last {reward_window_size} episodes): {average_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
