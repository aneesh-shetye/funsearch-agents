# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
import sys
import os
import logging

import concurrent.futures
import programs_database


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from funsearch import code_manipulation


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  #print('found a run_function') 
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]

initial_code_hopper = """
#Build an agent to solve an environment.  
#The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 

#The input to this agent  is the current state of the environment. Its output should be an action of the form (float, float, float) where each value ranges from -1 to 1.

#This output would represent torques applied on rotors such that: 
#action[0] = torque applied on the thigh rotor
#action[1] = torque applied on the leg rotor 
#action[2] = torque applied on teh foot rotor

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

def agent_v0(state) -> tuple[float, float, float]:
  #state[0] gives the z-coordinate of the torso (height of the hopper),
  #state[1] gives the angle of the torso, 
  #state[2] gives the angle of the thigh joint, 
  #state[3] gives the angle of the foot joint and, 
  #state[4] gives the velocity of the x-coordinate (height) of the torso
  #state[5] gives the velocity of the x-coordinate of the torso 
  #state[6] gives the velocity of the z-coordinate of the torso 
  #state[7] gives the angular velocity of the angle of the torso 
  #state[8] gives the angular velocity of the thigh hinge #state [9] gives the angular velocity of the leg hinge 
  #state[10] gives the angular velocity of the foot hinge 
  #Given the state output actions that would carry the object to the required position using the robotic arm.
                """

initial_code_ip = """

#Change the actions made by the agent.
#The environment is a cart with a pendulum on it and the agent's task is to keep it upright and not let it fall by pushing the cart by applying force to it.
#This input is a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart.
#Given this state, the agent returns an action which is the force applied to the cart which is a list with a single element, e.g. [0] where the element would range from [-3, 3]
#Improvement is measured by a "score" that is given by running your model in the environemnt. You do not have to implement. Look at the changes you made in the past and try to make  positive changes to increase this score.
#Make only small changes.
#Try to make the code short.
def agent_v0(state) -> list[float]:
  #The input is a 4 dimensional vector with  state[0] gives the position of the cart, state[1] gives the vertical angle of the pole on the cart, state[2] gives the linear velocity of the cart and state[3] gives the angular velocity of the pole on the cart.
  #Given this state, take an action which is a list with a single element, e.g. [0] where the element would range from [-3, 3]


                   """



def run(samplers, database, iterations: int = -1, ncpus: int = 2):
  """Launches a FunSearch experiment."""

  try:
    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    while iterations != 0:

      #with concurrent.futures.ThreadPoolExecutor(ncpus) as executor:
        # Use list comprehension to submit tasks to the thread pool
        #futures = [executor.submit(s.sample) for s in samplers]
        # Optionally, wait for all tasks to complete
        #concurrent.futures.wait(futures)
      database.recluster_islands()
      if len(database._islands) < 10: 
        for island_id in range(len(database._islands), 10): 
          scores_per_test = None
          prompt = programs_database.Prompt(code=initial_code_hopper, 
                                            version_generated=0,island_id=island_id) 
          #print(f'prompt in main: {prompt}')
          while not scores_per_test: 
            prompt = programs_database.Prompt(code=initial_code_hopper, 
                                            version_generated=0,island_id=island_id) 
            scores_per_test = samplers[0].sample(prompt, island_id)

      for island_id in range(len(database._islands)): 
        for s in samplers:
          s.sample(island_id=island_id)

      if iterations > 0:
        iterations -= 1
  except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Stopping.")
  database.backup()

