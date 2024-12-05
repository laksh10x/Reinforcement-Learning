import gymnasium as gym
import numpy as np

# Initialize the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# Q2.2: Collect data using a random policy
# Initialize variables
num_states = env.observation_space.n
num_actions = env.action_space.n

transition_counts = np.zeros((num_states, num_actions, num_states))
reward_sums = np.zeros((num_states, num_actions, num_states))
reward_counts = np.zeros((num_states, num_actions, num_states))

num_episodes = 1000

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Choose a random action
        action = env.action_space.sample()
        # Take the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update counts and rewards
        transition_counts[state, action, next_state] += 1
        reward_sums[state, action, next_state] += reward
        reward_counts[state, action, next_state] += 1

        # Move to the next state
        state = next_state

    if (episode + 1) % 100 == 0:
        print(f"Collected data for {episode + 1} episodes")

# Estimate T(s'|s,a) and R(s,a,s')
T = np.zeros((num_states, num_actions, num_states))
R = np.zeros((num_states, num_actions, num_states))

for s in range(num_states):
    for a in range(num_actions):
        total_transitions = np.sum(transition_counts[s, a, :])
        if total_transitions > 0:
            T[s, a, :] = transition_counts[s, a, :] / total_transitions
        for s_prime in range(num_states):
            if reward_counts[s, a, s_prime] > 0:
                R[s, a, s_prime] = reward_sums[s, a, s_prime] / reward_counts[s, a, s_prime]
            else:
                R[s, a, s_prime] = 0

# Q2.3: Implement value iteration
gamma = 0.9
theta = 1e-6  # Convergence threshold
V = np.zeros(num_states)

iteration = 0
while True:
    delta = 0
    for s in range(num_states):
        v = V[s]
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            action_value = 0
            for s_prime in range(num_states):
                action_value += T[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
            action_values[a] = action_value
        V[s] = max(action_values)
        delta = max(delta, abs(v - V[s]))
    iteration += 1
    if delta < theta:
        print(f"Value iteration converged after {iteration} iterations.")
        break
    if iteration % 10 == 0:
        print(f"Value iteration progress: {iteration} iterations")

# Q2.4: Extract the optimal policy
policy = np.zeros(num_states, dtype=int)

for s in range(num_states):
    action_values = np.zeros(num_actions)
    for a in range(num_actions):
        action_value = 0
        for s_prime in range(num_states):
            action_value += T[s, a, s_prime] * (R[s, a, s_prime] + gamma * V[s_prime])
        action_values[a] = action_value
    best_action = np.argmax(action_values)
    policy[s] = best_action

# Q2.5: Execute the optimal policy
test_episodes = 10
successes = 0

for episode in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

        # Optionally render the environment
        # env.render()
    if total_reward > 0:
        successes += 1
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

print(f"Success rate over {test_episodes} episodes: {successes / test_episodes * 100}%")
