import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# Create the Hopper environment
env_name = "Hopper-v4"
env = gym.make(env_name)

# Wrap the environment to make it compatible with Stable-Baselines3
env = DummyVecEnv([lambda: env])

# Instantiate the PPO agent
model = PPO(
    "MlpPolicy",       # Use a Multi-Layer Perceptron policy
    env,               # The environment
    verbose=1,         # Verbose output for training
    tensorboard_log="./ppo_hopper_tensorboard/"  # Log directory for TensorBoard
)

# Train the agent
timesteps = 1_000_000  # Set the number of training timesteps
print("Training the PPO agent...")
model.learn(total_timesteps=timesteps)

# Save the trained model
model.save("ppo_hopper")

# Load the trained model
model = PPO.load("ppo_hopper", env=env)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Visualize the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()

# Plot learning curve (optional)
rewards, timesteps = [], []
for i, log in enumerate(model.ep_info_buffer):
    rewards.append(log["r"])
    timesteps.append(log["l"])

plt.plot(range(len(rewards)), rewards)
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

