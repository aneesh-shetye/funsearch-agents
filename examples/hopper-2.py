
"""
#Build an agent to solve an environment.  
#This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.
#The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

#The function you are designing would take in a "state" argument which is a 11 dimensional vector:
#state[0] gives the z-coordinate of the torso (height of the hopper),
#state[1] gives the angle of the torso, 
#state[2] gives the angle of the thigh joint, 
#state[3] gives the angle of the foot joint and, 
#state[4] gives the velocity of the x-coordinate (height) of the torso
#state[5] gives the velocity of the x-coordinate of the torso 
#state[6] gives the velocity of the z-coordinate of the torso 
#state[7] gives the angular velocity of the angle of the torso 
#state[8] gives the angular velocity of the thigh hinge 
#state [9] gives the angular velocity of the leg hinge 
#state[10] gives the angular velocity of the foot hinge 


Complete the code for the agent that solves the environment. 
"""
import itertools

import random
import numpy as np

import funsearch

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

@funsearch.run
def evaluate(dummy: int):
  """Returns the total reward earned by the agent in the environment"""

  env = gym.make('LunarLander-v2')
  #env = TimeLimit(env, max_episode_steps=500) 
  observation, info = env.reset(seed=42)
  total_reward = 0 
  done = False
  truncated = False
  while not done and not truncated:
    action = agent(observation)
    observation, reward, done, truncated, info = env.step(action) 
    total_reward += reward
  print(f'total reward: {total_reward}')

  return int(total_reward)

@funsearch.evolve
def agent(state)->int:


