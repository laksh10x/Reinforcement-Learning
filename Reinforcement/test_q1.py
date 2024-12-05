import gymnasium as gym

# Initialize the environment with render_mode
env = gym.make("Blackjack-v1", render_mode="human")

# Reset the environment and obtain initial observation and info
observation, info = env.reset()

# Loop through 50 steps
for _ in range(50):
    action = env.action_space.sample()  # Sample a random action from the action space
    
    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Check if the episode is terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()  # Reset the environment

# Close the environment after completing the loop
env.close()
