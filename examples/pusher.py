"""
Build an agent to solve  gym's pusher-v4 environment.  
“Pusher” is a multi-jointed robot arm that is very similar to a human arm. The goal is to move a target cylinder (called object) to a goal position using the robot’s end effector (called fingertip). The robot consists of shoulder, elbow, forearm and wrist joints.

The input to this agent  is the current state of gym's pusher v-4 environment. Its output should be an action of the form (float, float, float, float, float, float, float) where each value ranges from -2 to 2.
This output would represent the torques applied by you to the hinge joints. Where:  
action[0] = Torque for the rotation of the panning the shoulder (for rotating in the x-y plane) 
action[1] = Torque for rotation of the shoulder lifting joint
action[2] = Rotation of the shoulder rolling joint
action[3] = Rotation of hinge joint that flexed the elbow 
action[4] = Rotation of hinge that rolls the forearm 
action[5] = Rotation of flexing the wrist 
action[6] = Rotation of rolling the wrist

The function you are designing would take in a "state" argument which is a 23 dimensional vector:
state[0] = Rotation of the panning the shoulder 
state[1] = Rotation of the shoulder lifting joint
state[2] = Rotation of the shoulder rolling joint
state[3] = Rotation of hinge joint that flexed the elbow
state[4] = Rotation of hinge that rolls the forearm
state[5] = Rotation of flexing the wrist
state[6] = Rotation of rolling the wrist
state[7] = Rotational velocity of the panning the shoulder
state[8] = Rotational velocity of the shoulder lifting joint 
state [9] = Rotational velocity of the shoulder rolling joint 
state[10] = Rotational velocity of hinge joint that flexed the elbow
state[11] = Rotational velocity of hinge that rolls the forearm
state[12] = Rotational velocity of flexing the wrist
state[13] = Rotational velocity of rolling the wrist 
state[14] = x-coordinate of the fingertip of the pusher 
state[15] = y-coordinate of the fingertip of the pusher 
state[16] = z-coordinate of the fingertip of the pusher 
state[17] = x-coordinate of the object to be moved
state[18] = y-coordinate of the object to be moved
state[19] = z-coordinate of the object to be moved 
state[20] = x-coordinate of the goal position of the object 
state[21] = y-coordinate of the goal position of the object
state[22] = z-coordinate of the goal position of the object

You have to maximize the reward which is calculated as:  
The total reward is: reward = reward_dist + reward_ctrl + reward_near.
reward_near: This reward is a measure of how far the fingertip of the pusher (the unattached end) is from the object, with a more negative value assigned for when the pusher’s fingertip is further away from the target. 
reward_dist: This reward is a measure of how far the object is from the target goal position, with a more negative value assigned if the object is further away from the target.
reward_control: A negative reward to penalize the pusher for taking actions that are too large. It is measured as the negative squared Euclidean norm of the action, i.e. as
.
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

  env = gym.make('Pusher-v4')
  env = TimeLimit(env, max_episode_steps=1000) 
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
def agent(state) -> tuple[float, float, float]:
  """
  The function you are designing would take in a "state" argument which is a 23 dimensional vector:
  state[0] = Rotation of the panning the shoulder 
  state[1] = Rotation of the shoulder lifting joint
  state[2] = Rotation of the shoulder rolling joint
  state[3] = Rotation of hinge joint that flexed the elbow
  state[4] = Rotation of hinge that rolls the forearm
  state[5] = Rotation of flexing the wrist
  state[6] = Rotation of rolling the wrist
  state[7] = Rotational velocity of the panning the shoulder
  state[8] = Rotational velocity of the shoulder lifting joint 
  state [9] = Rotational velocity of the shoulder rolling joint 
  state[10] = Rotational velocity of hinge joint that flexed the elbow
  state[11] = Rotational velocity of hinge that rolls the forearm
  state[12] = Rotational velocity of flexing the wrist
  state[13] = Rotational velocity of rolling the wrist 
  state[14] = x-coordinate of the fingertip of the pusher 
  state[15] = y-coordinate of the fingertip of the pusher 
  state[16] = z-coordinate of the fingertip of the pusher 
  state[17] = x-coordinate of the object to be moved
  state[18] = y-coordinate of the object to be moved
  state[19] = z-coordinate of the object to be moved 
  state[20] = x-coordinate of the goal position of the object 
  state[21] = y-coordinate of the goal position of the object
  state[22] = z-coordinate of the goal position of the object
  Complete the code for the agent that solves the environment. 
  The output of this function should be an action of the form (float, float, float, float, float, float, float) where each value ranges from -2 to 2.
  This output would represent the torques applied by you to the hinge joints. Where:  
  action[0] = Rotation of the panning the shoulder 
  action[1] = Rotation of the shoulder lifting joint
  action[2] = Rotation of the shoulder rolling joint
  action[3] = Rotation of hinge joint that flexed the elbow 
  action[4] = Rotation of hinge that rolls the forearm 
  action[5] = Rotation of flexing the wrist 
  action[6] = Rotation of rolling the wrist

  Given the state output actions that would carry the object to the required position using the robotic arm.
  """
  #return (0.9, 0.0, 0.0) 


