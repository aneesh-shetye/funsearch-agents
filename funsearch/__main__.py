import json
import logging
import os
import pathlib
import pickle
import time

from typing import List, Optional
import click
# import llm
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
#from llama import Dialog, Llama
from dotenv import load_dotenv

import wandb 

import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses

SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


def parse_input(filename_or_data):
  if len(filename_or_data) == 0:
    raise Exception("No input data specified")
  p = pathlib.Path(filename_or_data)
  if p.exists():
    if p.name.endswith(".json"):
      return json.load(open(filename_or_data, "r"))
    if p.name.endswith(".pickle"):
      return pickle.load(open(filename_or_data, "rb"))
    raise Exception("Unknown file format or filename")
  if "," not in filename_or_data:
    data = [filename_or_data]
  else:
    data = filename_or_data.split(",")
  if data[0].isnumeric():
    f = int if data[0].isdecimal() else float
    data = [f(v) for v in data]
  return data

@click.group()
@click.pass_context
def main(ctx):
  pass


@main.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument('inputs')
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
@click.option('--output_path', default="data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=15, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=2, type=click.INT, help='Samplers')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
def run(spec_file, inputs, model_name, output_path, load_backup, iterations, samplers, sandbox_type):
  """ Execute function-search algorithm:

\b
  SPEC_FILE is a python module that provides the basis of the LLM prompt as
            well as the evaluation metric.
            See examples/cap_set_spec.py for an example.\n
\b
  INPUTS    input filename ending in .json or .pickle, or a comma-separated
            input data. The files are expected contain a list with at least
            one element. Elements shall be passed to the solve() method
            one by one. Examples
              8
              8,9,10
              ./examples/cap_set_input_data.json
"""

  # Load environment variables from .env file.
  #
  # Using OpenAI APIs with 'llm' package requires setting the variable
  # OPENAI_API_KEY=sk-...
  # See 'llm' package on how to use other providers.
  load_dotenv()

  timestamp = str(int(time.time()))
  log_path = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    #logging.info(f"Writing logs to {log_path}")

  # model = llm.get_model(model_name)
  # model.key = model.get_key()

  checkpoint = '/vast/work/public/ml-datasets/llama-3.1/Meta-Llama-3.1-8B-Instruct/'
  tokenizer_path = '/vast/work/public/ml-datasets/llama-3.1/Meta-Llama-3.1-8B-Instruct/tokenizer.model'

  model_id = "meta-llama/Llama-3.3-70B-Instruct"
  #model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  #model_id = "upiter/TinyCodeLM-400M"

  pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
  )

  '''
  model_id_meta = 'meta-llama/Meta-Llama-3.2-1B-Instruct'

  pipeline2 = transformers.pipeline(
    "text-generation",
    model=model_id_meta,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
  )
  '''

  '''
  generator = Llama.build(
          ckpt_dir=checkpoint,
          tokenizer_path=tokenizer_path,
          max_seq_len=512,
          max_batch_size=6)

  '''
  model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)


  #pipeline2 = None
  lm = sampler.LLM(1, generator=pipeline, log_path=log_path)

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  conf = config.Config(num_evaluators=4)
  database = programs_database.ProgramsDatabase(
    conf.programs_database, template, function_to_evolve, identifier=timestamp) 
  if load_backup:
    database.load(load_backup)

  inputs = parse_input(inputs)
  #inputs = None

  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
  evaluators = [evaluator.Evaluator(
    database,
    sandbox_class(base_path=log_path),
    template,
    function_to_evolve,
    function_to_run,
    inputs,
  ) for _ in range(conf.num_evaluators)]

  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body

  samplers = [sampler.Sampler(database, evaluators, lm)
              for _ in range(2)]

  #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  #hardcoded the num islands
  #print(f'initial:{initial}')
  initial_code_pusher = """
#Build an agent to solve  gym's pusher-v4 environment.  
#“Pusher” is a multi-jointed robot arm that is very similar to a human arm. The goal is to move a target cylinder (called object) to a goal position using the robot’s end effector (called fingertip). The robot consists of shoulder, elbow, forearm and wrist joints.

#The input to this agent  is the current state of gym's pusher v-4 environment. Its output should be an action of the form (float, float, float, float, float, float, float) where each value ranges from -2 to 2.
#This output would represent the torques applied by you to the hinge joints. Where:  
#action[0] = Rotation of the panning the shoulder 
#action[1] = Rotation of the shoulder lifting joint
#action[2] = Rotation of the shoulder rolling joint
#action[3] = Rotation of hinge joint that flexed the elbow 
#action[4] = Rotation of hinge that rolls the forearm 
#action[5] = Rotation of flexing the wrist 
#action[6] = Rotation of rolling the wrist

#The function you are designing would take in a "state" argument which is a 23 dimensional vector:
#state[0] = Rotation of the panning the shoulder 
#state[1] = Rotation of the shoulder lifting joint
#state[2] = Rotation of the shoulder rolling joint
#state[3] = Rotation of hinge joint that flexed the elbow
#state[4] = Rotation of hinge that rolls the forearm
#state[5] = Rotation of flexing the wrist
#state[6] = Rotation of rolling the wrist
#state[7] = Rotational velocity of the panning the shoulder
#state[8] = Rotational velocity of the shoulder lifting joint 
#state [9] = Rotational velocity of the shoulder rolling joint 
#state[10] = Rotational velocity of hinge joint that flexed the elbow
#state[11] = Rotational velocity of hinge that rolls the forearm
#state[12] = Rotational velocity of flexing the wrist
#state[13] = Rotational velocity of rolling the wrist 
#state[14] = x-coordinate of the fingertip of the pusher 
#state[15] = y-coordinate of the fingertip of the pusher 
#state[16] = z-coordinate of the fingertip of the pusher 
#state[17] = x-coordinate of the object to be moved
#state[18] = y-coordinate of the object to be moved
#state[19] = z-coordinate of the object to be moved 
#state[20] = x-coordinate of the goal position of the object 
#state[21] = y-coordinate of the goal position of the object
#state[22] = z-coordinate of the goal position of the object
#Complete the code for the agent that solves the environment that maximizes the reward given as: 
#reward = reward_dist + reward_ctrl + reward_near.
#reward_near: This reward is a measure of how far the fingertip of the pusher (the unattached end) is from the object, with a more negative value assigned for when the pusher’s fingertip is further away from the target.  
#reward_dist: This reward is a measure of how far the object is from the target goal position, with a more negative value assigned if the object is further away from the target. 
#reward_control: A negative reward to penalize the pusher for taking actions that are too large. It is measured as the negative squared Euclidean norm of the action, i.e. as
def agent_v0(state) -> tuple[float, float, float, float, float, float, float]:
  #state[0] = Rotation of the panning the shoulder 
  #state[1] = Rotation of the shoulder lifting joint
  #state[2] = Rotation of the shoulder rolling joint
  #state[3] = Rotation of hinge joint that flexed the elbow
  #state[4] = Rotation of hinge that rolls the forearm
  #state[5] = Rotation of flexing the wrist
  #state[6] = Rotation of rolling the wrist
  #state[7] = Rotational velocity of the panning the shoulder
  #state[8] = Rotational velocity of the shoulder lifting joint 
  #state [9] = Rotational velocity of the shoulder rolling joint 
  #state[10] = Rotational velocity of hinge joint that flexed the elbow
  #state[11] = Rotational velocity of hinge that rolls the forearm
  #state[12] = Rotational velocity of flexing the wrist
  #state[13] = Rotational velocity of rolling the wrist 
  #state[14] = x-coordinate of the fingertip of the pusher 
  #state[15] = y-coordinate of the fingertip of the pusher 
  #state[16] = z-coordinate of the fingertip of the pusher 
  #state[17] = x-coordinate of the object to be moved
  #state[18] = y-coordinate of the object to be moved
  #state[19] = z-coordinate of the object to be moved 
  #state[20] = x-coordinate of the goal position of the object 
  #state[21] = y-coordinate of the goal position of the object
  #state[22] = z-coordinate of the goal position of the object
  #Given the state output actions that would carry the object to the required position using the robotic arm.
                """


  initial_code_hopper = """
#Build an agent to solve an environment.  
#Just give the code, do not provide the chain of thought or any comments. 
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
#Just provide an improvement over the following function, do not provide any explaination, chain-of-thought-reasoning or any comments. 

#Use template controllers given below to build the agent/function.Do not use anything else  

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

def agent_v0(state) -> tuple[float, float, float]:
  return (0.9, 0.9, 0.9) 

#use the templates above to form the code. Do not use anything else
                """

  initial_code_lander = """
#Build an agent to solve an environment.  
#This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.
#The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

def agent_v0(state)-> int:
"""

  initial_code = """
#Build an agent to solve an environment.  
#The  environment has a hopper - a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts. 

#The input to this agent  is the current state of the environment. Its output should be an action of the form (float, float, float) where each value ranges from -1 to 1.

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

  for island_id in range(10):
    scores_per_test = None
    prompt = programs_database.Prompt(code=initial_code_hopper, version_generated=0, 
                                      island_id=island_id) 

    print(f'prompt in main: {prompt}')
    while not scores_per_test: 
      scores_per_test = samplers[0].sample(prompt, island_id=island_id)



  '''
  evaluators[0].analyse(initial, island_id=None, version_generated=None)
  assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                   "See e.g. the error files under sandbox data.")
  '''

  #print(f'samplers: {type(samplers)}') 
  core.run(samplers, database, iterations, conf.ncpus)


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
  """List programs from a stored database (usually in data/backups/ )"""
  conf = config.Config(num_evaluators=4)

  # A bit silly way to list programs. This probably does not work if config has changed any way
  database = programs_database.ProgramsDatabase(
    conf.programs_database, None, "", identifier="")
  database.load(db_file)

  progs = database.get_best_programs_per_island()
  '''
  print("BEST PROGS PER ISLAND:") 
  print(progs) 
  print("Found {len(progs)} programs")
  '''


if __name__ == '__main__':

  '''
  wandb.init(project='funsearch', 
             config=config.Config(num_evaluators=4), 
             settings=wandb.Settings(start_method="fork"), 
             reinit=True)
  '''
  main()
  #wandb.finish()


