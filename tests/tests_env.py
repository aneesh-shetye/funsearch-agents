import itertools

import random 
import numpy as np


import gymnasium as gym
from gym.wrappers import TimeLimit

def evaluate(dummy: int) -> float:
  """Returns the total reward earned by the agent in the environment"""
  env = gym.make('InvertedPendulum-v4')
  #env = TimeLimit(env, max_episode_steps=500)
  observation, info = env.reset()
  total_reward = 0 
  done = False
  truncated = False
  while not done and not truncated: 
    action = agent(observation)  
    observation, reward, done, truncated, info = env.step(action) 
    total_reward += reward
  print(f'total reward: {total_reward}')

  return total_reward


def agent(state) -> list[float]:
  """
  The input to this function is the current state ofs inverted pendulum environment.
  Its a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart.
  Given this state, take an action which is a list with a single element, e.g. [0] where the element would range from [-3, 3]
  """
  return [0]

