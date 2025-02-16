
"""
Build an agent to solve gym's reacher-v4 environment. The agent is a two-jointed robot arm. The goal is to  move the robot's end effector (called fingertip) close to a target that is spawned at a random position.  
The input to this agent  is the current state of gym's reacher-v4 environment. Its output should be of the form (float, float) where each value ranges from -1 to 1.The action represents the torques applied at the hinge joints. 
The function you are designing would take in a "state" argument which is a 10 dimensional vector:
state[0] gives the cosine of the angle of the first arm.  
state[1] gives the cosine of the angle of the second arm.  
state[2] gives the sine of the angle of the first arm 
state[3] gives the sine of the angle of the second arm 
state[4] gives the x-coordinate of the target 
state[5] gives the y-coordinate of the target 
state[6] gives the angular velocity of the first arm  state[7] gives the angular velocity of the second arm 
state[7] gives the angular velocity of the second arm.
state[8] gives the x-value of position of position_fingertip-position_target 
state[9] gives the y-value of position of position_fingertip-position_target 
state[10] gives the z-value of position of position_fingertip-position_target 

Complete the code for the agent that solves the environment. Make as small changes as possible. 
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

  env = gym.make('Reacher-v4')
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

  return int(total_reward)

@funsearch.evolve
def agent(state) -> tuple[float, float]:
  """
The function you are designing takes in a "state" argument which is a 10 dimensional vector:
state[0] gives the cosine of the angle of the first arm.  
state[1] gives the cosine of the angle of the second arm.  
state[2] gives the sine of the angle of the first arm 
state[3] gives the sine of the angle of the second arm 
state[4] gives the x-coordinate of the target 
state[5] gives the y-coordinate of the target 
state[6] gives the angular velocity of the first arm  
state[7] gives the angular velocity of the second arm 
state[8] gives the x-value of position of position_fingertip-position_target 
state[9] gives the y-value of position of position_fingertip-position_target 
state[10] gives the z-value of position of position_fingertip-position_target 
  """
  return (0, 0) 


