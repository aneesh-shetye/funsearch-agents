import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PIL import Image
import os

# Create the Hopper environment
env_name = "Hopper-v4"
env = gym.make(env_name, render_mode='rgb_array')

# Wrap the environment
env = DummyVecEnv([lambda: env])

# Load the trained model
model = PPO.load("ppo_hopper", env=env)

# Directory to save images
image_dir = "state_images"
os.makedirs(image_dir, exist_ok=True)

# Visualize the trained agent and save images
obs = env.reset()
frame_idx = 0

for _ in range(1000):
    # Get action from the model
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    # Render and capture the frame as an image
    frame = env.render()  # Render as an RGB array
    print(frame) 
    image = Image.fromarray(frame)
    
    # Save the frame as an image file
    image.save(os.path.join(image_dir, f"frame_{frame_idx:04d}.png"))
    frame_idx += 1

    if dones:
        break

env.close()

print(f"Images saved in directory: {image_dir}")

