import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PIL import Image
import os

# Create the Hopper environment
env_name = "MountainCarContinuous-v0"
env = gym.make(env_name, render_mode='rgb_array')

# Wrap the environment
#env = DummyVecEnv([lambda: env])

# Load the trained model
#model = PPO.load("ppo_hopper", env=env)

# Directory to save images
image_dir = "state_images"
os.makedirs(image_dir, exist_ok=True)

# Visualize the trained agent and save images
obs = env.reset()
frame_idx = 0

done = False
truncation = False
while not done or not truncation:
    # Get action from the model
    #action, _states =  
    action = env.action_space.sample()
    out = env.step(action)
    print(out)
    obs, rewards, done, truncation, info = env.step(action)

    # Render and capture the frame as an image
    frame = env.render()  
    image = Image.fromarray(frame)
    
    # Save the frame as an image file
    image.save(os.path.join(image_dir, f"frame_{frame_idx:04d}.png"))
    frame_idx += 1
env.close()

print(f"Images saved in directory: {image_dir}")

