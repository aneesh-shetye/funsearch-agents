"""
Change the agent make a policy that maximizes rewards in the environment. 
The environment is gym's cartpole environment.  
The input to this agent is the current state of gym's cartpole v-1 environment. This input is a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart. 
Given this state, the agent returns an action which is either 0 or 1. 
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

  env = gym.make('CartPole-v1')
  observation, info = env.reset()
  total_reward = 0 
  done = False
  while not done: 
    action = agent(observation)  
    observation, reward, done, truncated, info = env.step(action) 
    total_reward += reward
  print(f'total reward: {total_reward}')

  return total_reward

#def solve()
#find the most recent agent and run it in the environment. Return "score"




@funsearch.evolve
def agent(state) -> float:
  """
  The input to this function is the current state of gym's cartpole v-1 environment. 
  Its a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart. 
  Given this state, take an action which is either 0 or 1. 
  """
  return 0 


