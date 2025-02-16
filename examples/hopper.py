
"""
Build an agent to solve  gym's hopper-v4 environment. The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 
The input to this agent  is the current state of gym's hopper v-4 environment. Its output should be of the form (float, float, float) where each value ranges from -1 to 1.
Build the function using templates of PID, LQR and LPG controllers giveen below. 
The function you are designing would take in a "state" argument which is a 11 dimensional vector:
state[0] gives the z-coordinate of the torso (height of the hopper),
state[1] gives the angle of the torso, 
state[2] gives the angle of the thigh joint, 
state[3] gives the angle of the foot joint and, 
state[4] gives the velocity of the x-coordinate (height) of the torso
state[5] gives the velocity of the x-coordinate of the torso 
state[6] gives the velocity of the z-coordinate of the torso 
state[7] gives the angular velocity of the angle of the torso 
state[8] gives the angular velocity of the thigh hinge 
state [9] gives the angular velocity of the leg hinge 
state[10] gives the angular velocity of the foot hinge 


and the output should be an action of the form (float, float, float) where each value ranges from -1 to 1. 
This output would represent torques applied on rotors such that: 
action[0] = torque applied on the thigh rotor
action[1] = torque applied on the leg rotor 
action[2] = torque applied on teh foot rotor

Complete the code for the agent that solves the environment. 
#Just provide an improvement over the following function, do not provide any explaination, chain-of-thought-reasoning or any comments. 
Only compose the templates given below to form the funtion. 
"""
import itertools

import random
import numpy as np

import funsearch

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

from scipy import signal
from scipy.linalg import solve_continuous_are

# -----------------------------
# PID Controller Template
# -----------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# -----------------------------
# LQR Controller Template
# -----------------------------
def lqr(A, B, Q, R):
    # Solve the Continuous Algebraic Riccati Equation (CARE)
    P = solve_continuous_are(A, B, Q, R)
    # Compute the LQR gain
    K = np.linalg.inv(R) @ (B.T @ P)
    return K

# -----------------------------
# LQG Controller Template
# -----------------------------
def kalman_filter(A, C, W, V):
    # W: process noise covariance, V: measurement noise covariance
    P = solve_continuous_are(A.T, C.T, W, V)
    L = P @ C.T @ np.linalg.inv(V)
    return L

def lqg(A, B, C, Q, R, W, V):
    # LQR for optimal control
    K = lqr(A, B, Q, R)
    # Kalman Filter for state estimation
    L = kalman_filter(A, C, W, V)
    return K, L

@funsearch.run
def evaluate(dummy: int):
  """Returns the total reward earned by the agent in the environment"""

  env = gym.make('Hopper-v4')
  #env = TimeLimit(env, max_episode_steps=1000) 
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
def agent(state) -> tuple[float, float, float]:
  return (0.9, 0.9, 0.9) 

