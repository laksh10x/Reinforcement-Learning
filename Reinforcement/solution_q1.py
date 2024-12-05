import gymnasium as gym
import numpy as np
import random

# Initialize the Blackjack environment
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 1.0    # Exploration rate
epsilon_min = 0.05
epsilon_decay = 0.999995
num_episodes = 500000

# Initialize Q-table
Q = {}

# Function to get Q-value for a state-action pair
def get_Q(state, action):
    return Q.get((state, action), 0.0)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            # Exploit
            q_values = [get_Q(state, a) for a in range(env.action_space.n)]
            action = np.argmax(q_values)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax([get_Q(next_state, a) for a in range(env.action_space.n)])
        td_target = reward + gamma * get_Q(next_state, best_next_action) * (not done)
        td_delta = td_target - get_Q(state, action)
        Q[(state, action)] = get_Q(state, action) + alpha * td_delta

        state = next_state

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Optional: Print progress every 10000 episodes
    if (episode + 1) % 10000 == 0:
        print(f"Episode {episode + 1}/{num_episodes}")

# Extract the optimal policy
policy = {}
for state in Q.keys():
    actions = [get_Q(state[0], a) for a in range(env.action_space.n)]
    best_action = np.argmax(actions)
    policy[state[0]] = best_action

# Testing the trained agent
num_test_episodes = 10000
wins = 0
ties = 0
losses = 0

for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Choose the best action
        q_values = [get_Q(state, a) for a in range(env.action_space.n)]
        action = np.argmax(q_values)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

    if reward > 0:
        wins += 1
    elif reward == 0:
        ties += 1
    else:
        losses += 1

print(f"Out of {num_test_episodes} games:")
print(f"Wins: {wins}")
print(f"Ties: {ties}")
print(f"Losses: {losses}")
print(f"Win rate: {wins / num_test_episodes * 100}%")
