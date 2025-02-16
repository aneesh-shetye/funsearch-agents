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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import os 
# import llm
import numpy as np


import evaluator
import programs_database
import concurrent.futures

from transformers import AutoTokenizer

import wandb 

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True)


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, generator, 
               meta_gen=None, log_path=None) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.tokenizer = tokenizer
    #self.model = model
    self.generator = generator
    self.meta_gen = meta_gen
    self.prompt_count = 0
    self.log_path = log_path

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    #response = self.generator.generate(prompt)
    #print("###################################")
    #print(f"PROMPT: {prompt}")
    '''
    response = self.model.generate(**self.tokenizer(prompt, 
                          return_tensors='pt', padding=True,
                          truncation=True),max_length=300 )
    result = self.generator.text_completion([prompt], max_gen_len=512, 
                                         temperature=0.6, top_p=0.9) 
    '''

    #print("#####################################") 
    #print(f'prompt: {prompt}')
    meta_prompt = f"this is an agent trying to solve the" 


    '''
    prompt = self.meta_gen(meta_prompt, max_new_tokens=1024, temperature=1) 
    prompt = prompt[0]['generated_text']
    '''

    #print(f'prompt: {prompt}')
    #inputs = tokenizer(prompt, return_tensors="pt").to(self.generator.device) 
    result = self.generator(prompt, max_new_tokens=1024)
    #response = tokenizer.decode(result[0])#['generated_text']
    #print(f'result: {result}') 
    response = result[0]['generated_text']
    #response = self.tokenizer.batch_decode(response, 
                                           #skip_special_tokens=True)
    self._log(prompt, response, self.prompt_count)
    self.prompt_count += 1
    return response

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    print(f'drawing samples with following prompt: {prompt}') 
    samples = [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
    return samples

  def _log(self, prompt: str, response: str, index: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}.log", "a") as f:
        f.write(prompt)
      with open(self.log_path / f"response_{index}.log", "a") as f:
        f.write(str(response))


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      model: LLM
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = model
    self.ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

  def process_sample(self, sample): 
    chosen_evaluator = np.random.choice(self._evaluators)
    return chosen_evaluator.analyse(sample, prompt.island_id, prompt.version_generated)

  def sample(self, prompt=None, island_id=None):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    print(f'quering prompt from island_id: {island_id}')
    if not prompt and not island_id:
      prompt = self._database.get_prompt()
    if island_id and not prompt: 
      prompt = self._database.get_prompt(island_id)

    print(f'prompt: {prompt.code}')
    if type(prompt) == str: 
      samples = self._llm.draw_samples(prompt)
      print(f'samples in sampler: {samples}') 
    else: 
      samples = self._llm.draw_samples(prompt.code)
    #print(f'samples in sampler: {samples}') 
    # This loop can be executed in parallel on remote evaluator machines.
    for sample in samples:
      #print(f'sample in sampler: {sample}') 
      chosen_evaluator = np.random.choice(self._evaluators)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=self.ncpus) as executor:
    # Submit tasks for all samples
      #futures = [executor.submit(process_sample, sample) for sample in samples]

    # Optionally collect results (if needed)
      #scores_per_test = [future.result() for future in concurrent.futures.as_completed(futures)]
      print(f'sampling from island_id: {prompt.island_id}')
      scores_per_test = chosen_evaluator.analyse(
          sample, prompt.island_id, prompt.version_generated)
      print(f'scores_per_test in sampler: {scores_per_test}')
      
    return scores_per_test



