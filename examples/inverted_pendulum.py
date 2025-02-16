"""
Change the actions made by the agent.
The environment is a cart with a pendulum on it and the agent's task is to keep it upright and not let it fall by pushing the cart by applying force to it.
This input is a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart.
Given this state, the agent returns an action which is the force applied to the cart which is a list with a single element, e.g. [0] where the element would range from [-3, 3]
Improvement is measured by a "score" that is given by running your model in the environemnt. You do not have to implement. Look at the changes you made in the past and try to make  positive changes to increase this score.
Make only small changes.
Try to make the code short.
"""
import itertools

import random 
import numpy as np

import funsearch

import gymnasium as gym

@funsearch.run
def evaluate(dummy: int) -> float:
  """Returns the total reward earned by the agent in the environment"""

  env = gym.make('InvertedPendulum-v4')
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

  return total_reward

@funsearch.evolve
def agent(state) -> list[float]:
  """
  The input is a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart.
  Given this state, take an action which is a list with a single element, e.g. [0] where the element would range from [-3, 3]
  """
  return [0]


