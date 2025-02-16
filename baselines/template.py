import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

# Step 1: Create and wrap the environment
env = gym.make('LunarLander-v2')  # Replace with the correct environment if needed
check_env(env)
#vec_env = make_vec_env(lambda: env, n_envs=1)

# Step 2: Create the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device='cpu'
)

# Step 3: Train the PPO model
timesteps = 100000  # Set the number of timesteps
model.learn(total_timesteps=timesteps)
vec_env = model.get_env()

# Step 4: Save the model
model.save("ppo_pendulum")

# Step 5: Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Step 6: Test the trained model
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    #env.render()
    if done:
        obs = env.reset()

obs = vec_env.reset()
total_reward = 0 
done = False
while not done: 
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action) 
    total_reward += reward  

print(f'total reward at the end of training: {total_reward}')

env.close()

