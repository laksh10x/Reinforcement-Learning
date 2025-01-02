Reinforcement Learning

Welcome to the Reinforcement Learning repository! 🚀 This repository showcases two exciting projects that explore reinforcement learning algorithms applied to classic games—Blackjack and Frozen Lake—using Python and the Gymnasium library. These implementations demonstrate the power of reinforcement learning techniques such as Q-Learning and Value Iteration to create intelligent agents capable of optimizing their strategies.

Blackjack AI 🃏🧠

About the Game

Blackjack is one of the most popular casino games in the world, where the objective is to have a hand value closer to 21 than the dealer without exceeding it. In this project, we simulate the Blackjack environment and train an AI agent to learn optimal strategies using Q-Learning, a model-free reinforcement learning algorithm.

Features

Simulated Blackjack environment using Gymnasium's Blackjack-v1.

Q-Learning Implementation: The AI agent iteratively updates its action-value (Q) table to determine the best moves based on rewards.

Real-time interaction with the environment, including GUI visualization for gameplay.

Optimal decision-making based on learned strategies to maximize rewards.

Technologies Used

Python 3.9

Gymnasium library for game simulation

Q-Learning algorithm for reinforcement learning

Frozen Lake AI 🌨️🧩

About the Game

Frozen Lake is a grid-based puzzle game that challenges the player to navigate a slippery frozen lake and safely reach the goal without falling into holes. Fans of the classic Pokémon GameBoy games might find this puzzle familiar, as it resembles the slippery-floor puzzles used in those adventures! ❄️

In this project, we approach the problem using model-based reinforcement learning techniques to uncover the optimal policy for navigation.

Features

Simulated Frozen Lake environment using Gymnasium's FrozenLake-v1.

Random Policy Exploration: Collects training data over 1000 episodes to estimate transition and reward functions.

Value Iteration: Computes the optimal value function based on learned transition probabilities and rewards.

Policy Extraction: Derives the optimal policy based on the learned value function.

Visualization of gameplay with dynamic decision-making based on the extracted policy.

Technologies Used

Python 3.9

Gymnasium library for game simulation

Value Iteration and Policy Extraction for optimal decision-making

Repository Contents

solution_q1.py - Implementation of the Blackjack AI using Q-Learning.

solution_q2.py - Implementation of the Frozen Lake AI using Value Iteration and Policy Extraction.

test_q1.py - Script to validate the setup of the Blackjack environment.

test_q2.py - Script to validate the setup of the Frozen Lake environment.

How to Run 🖥️

Clone the repository:

git clone https://github.com/laksh10x/Reinforcement-Learning.git

Install dependencies:

pip install gymnasium

Execute the scripts:

python solution_q1.py   # For Blackjack AI
python solution_q2.py   # For Frozen Lake AI

Demo and Visualizations 🎥

Both implementations include interactive visualizations, so you can see the AI agent making decisions in real-time as it plays the games. Watch the Blackjack agent take calculated risks and the Frozen Lake agent navigate the slippery maze with precision.

Why Reinforcement Learning? 🤖

Reinforcement learning is a key subfield of AI that enables agents to make sequential decisions through exploration and learning from experience. This repository provides practical examples of how reinforcement learning techniques can be applied to solve real-world problems modeled through games.
