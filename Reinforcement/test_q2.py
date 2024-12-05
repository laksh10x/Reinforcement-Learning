import gymnasium as gym

# Initialize the FrozenLake environment
env = gym.make(
    "FrozenLake-v1",
    desc=None,
    map_name="4x4",
    render_mode="human",  # Fixed quotation marks
    is_slippery=True
)

# Reset the environment to get the initial observation and info
observation, info = env.reset()

# Run a loop for 50 steps
for _ in range(50):
    action = env.action_space.sample()  # Sample a random action
    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Check if the episode is terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()  # Reset the environment

# Close the environment after the loop
env.close()
