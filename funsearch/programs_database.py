# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
import pathlib
import pickle
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Iterable, Tuple

import seaborn as sns
import random
from absl import logging
import numpy as np
import scipy
from sklearn.manifold import MDS 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import eigh  # For eigen decomposition

import torch

import code_manipulation
import config as config_lib

import gymnasium as gym 

import wandb
from threading import Lock

import transformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


#wandb_lock = Lock()

#model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"



Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]

def create_clusters(self):

  def dbscan(similarity_matrix, eps):

    distance_matrix = (1 - similarity_matrix).clamp(min=0)

    dbscan = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = dbscan.fit_predict(distance_matrix)

    return labels

    embeddings = [agent['embedding'] for agent in self.db]
    similariy_score = self.embedding_model.model.similarity(embeddings, embeddings)
    cluster_labels = dbscan(similariy_score, 0.05)
        # for each cluster find the highest scoring element only
        # keep that in the cluster
        # each cluster will have a saturated or not tag

    for cluster_label in set(cluster_labels):
      highest_score = 0
      highest_idx = 0
      for idx, label in enumerate(cluster_labels):
        if label==cluster_label and self.db[idx]['score']>highest_score:
          highest_score = self.db[idx]['score']
          highest_idx = idx
          self.cluster.append({
                "code": self.db[highest_idx]['code'], 
                "score": self.db[highest_idx]['score'], 
                "embedding": self.db[highest_idx]['embedding'],
                "saturated": False})
          print(f"Added agent to cluster {len(self.cluster)} with score {highest_score}.")

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  print(list(scores_per_test.keys())[-1])
  return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(int(scores_per_test[k]) for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
      identifier: str = "", 
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    '''
    self.tokenizer = tokenizer
    self.tokenizer.pad_token = tokenizer.eos_token
    '''
    #self.model = model
    self.model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    # Initialize empty islands.
    self._islands: list[Island] = []
    for _ in range(config.num_islands):
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period))
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    self._best_program_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)

    self._last_reset_time: float = time.time()
    self._program_counter = 0
    self._backups_done = 0
    self.identifier = identifier

  def get_best_programs_per_island(self) -> Iterable[Tuple[code_manipulation.Function | None]]:
    return sorted(zip(self._best_program_per_island, self._best_score_per_island), key=lambda t: t[1], reverse=True)

  def save(self, file):
    """Save database to a file"""
    data = {}
    keys = ["_islands", "_best_score_per_island", "_best_program_per_island", "_best_scores_per_test_per_island"]
    for key in keys:
      data[key] = getattr(self, key)
    pickle.dump(data, file)

  def load(self, file):
    """Load previously saved database"""
    data = pickle.load(file)
    for key in data.keys():
      setattr(self, key, data[key])

  def backup(self):
    filename = f"program_db_{self._function_to_evolve}_{self.identifier}_{self._backups_done}.pickle"
    p = pathlib.Path(self._config.backup_folder)
    if not p.exists():
      p.mkdir(parents=True, exist_ok=True)
    filepath = p / filename
    logging.info(f"Saving backup to {filepath}.")

    with open(filepath, mode="wb") as f:
      self.save(f)
    self._backups_done += 1

  def get_prompt(self, island_id=None) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    if not island_id: 
      island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_prompt()
    return Prompt(code, version_generated, island_id)

  def _register_program_in_island(
      self,
      program: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the specified island."""
    score = _reduce_score(scores_per_test)
    print(f'scores_per_test: {scores_per_test}')
    #with wandb_lock: 
    #wandb.log({f'score_island_{island_id}': score})
    #wandb.log({f'program_{island_id}':wandb.Html(f'score: {score}<pre>{program.__str__()}</pre>')})

    if score > self._best_score_per_island[island_id]:
      #print(f'type self._islands[island_id]: {type(self._islands[island_id])}')
      self._islands[island_id].register_program(program, scores_per_test)
      self._best_program_per_island[island_id] = program
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      self._best_score_per_island[island_id] = int(score)
      #with wandb_lock: 
      #wandb.log({f'best_score_island_{island_id}': self._best_score_per_island[island_id]})

      logging.info('Best score of island %d increased to %s', island_id, score)
      #with wandb_lock: 
      #wandb.log({f'best_program_{island_id}':wandb.Html(f'<pre>{program.__str__()}</pre>')})

      namespace = {}
      exec(program.__str__(), globals(), namespace)

      '''
      if 'agent' in namespace:
        try:
          self.sanity_check(namespace['agent'], island_id)
        except Exception as e:
          print(f"Sanity check failed with exception: {e}")
      else:
        print("Error: 'agent' function not found in the program.")
      '''

      #print(f'best program: {program.__str__()}')
      #print(f'vars(program): {vars(program)}') 

  def register_program(
      self,
      program: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores_per_test)
    else:
      self._register_program_in_island(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    if (time.time() - self._last_reset_time > self._config.reset_period):
      self._last_reset_time = time.time()
      self.reset_islands()

    # Backup every N iterations
    if self._program_counter > 0:
      self._program_counter += 1
      if self._program_counter > self._config.backup_period:
        self._program_counter = 0
        self.backup()

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      self._best_score_per_island[island_id] = -float('inf')
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      self._register_program_in_island(founder, island_id, founder_scores)


  def spectral_clustering(self, similarity_matrix, num_clusters):
      degree_matrix = np.diag(similarity_matrix.sum(axis=1))
      laplacian_matrix = degree_matrix - similarity_matrix

      eigvals, eigvecs = eigh(laplacian_matrix, subset_by_index=[0, num_clusters-1])

      kmeans = KMeans(n_clusters=num_clusters, random_state=0)
      clusters = kmeans.fit_predict(eigvecs)

      return clusters


  def recluster_islands(self) -> None: 

    scores = []
    #plt.figure(figsize=(8, 6))

    programs = [] 
    scores_per_test = [] 

    programs_for_embed = []
    for isle_id, island in enumerate(self._islands):
      program_embeds = []

      for signature in island._clusters.keys(): 
        implementations = [copy.deepcopy(i) for i in 
                              island._clusters[signature]._programs]
        implementations_for_embed = [str(copy.deepcopy(i.body)) for i in island._clusters[signature]._programs]
        num_programs = len(island._clusters[signature]._programs)
        print(num_programs)
        score = island._clusters[signature]._score
        island_scores = [score for i in range(num_programs)]
        scores.extend(island_scores)
        #print(implementations)
        programs.extend(implementations)
        programs_for_embed.extend(implementations_for_embed)
        scores_per_test.append(self._best_scores_per_test_per_island[isle_id])

    if len(programs) < 1: 
      print(f'len programs < 1:\n {programs}')
      return 0  
    print(f'len programs: {len(programs)}') 
    print(f'len scores: {len(scores)}') 
    print(f'scores in recluster: {scores}')
    #print(programs)
    embeddings = self.get_embeddings(programs_for_embed)
    print(f'embeddings.shape in recluster: {embeddings.shape}') 
    '''
    projector = MDS()
    proj = projector.fit_transform(embeddings) 
    x = proj[:,0]
    y = proj[:,1]

    plt.scatter(x, y)#, label=f'Island:{isle_id}') 

    plt.savefig('scatter_plot.png') 
    #wandb.log({'scatter_plot':wandb.Image('scatter_plot.png')})
    #dist = (1 - sim).clamp(min=0)
    '''

    similarity_score = self.model.similarity(embeddings, embeddings)
    plt.figure(figsize=(10,8))
    sns.heatmap(similarity_score, annot=False, cmap='viridis')
    plt.title("simialrity between programs")
    plt.savefig("heatmap.png")
    #wandb.log({"similarity heatmap": wandb.Image("heatmap.png")})
    distance_matrix = (1-similarity_score).clamp(min=0)
    print(f'similarity_score: {similarity_score}')
    print(f'distance_matrix: {distance_matrix}')
    dbscan = DBSCAN(eps=0.05, min_samples=2, metric="precomputed")
    labels = dbscan.fit_predict(distance_matrix)

    print('labels using distance matrix:') 
    print(labels)

    if -1 in labels: 
      labels += 1 

    for n, label in enumerate(labels): 
      self._islands[label] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      self._register_program_in_island(programs[n], label, scores_per_test[n])



  def get_embeddings(self,text: str): 
    embeddings = self.model.encode(text) 
    return embeddings



class Island:
  """A sub-population of the programs database."""

  def __init__(
      self,
      template: code_manipulation.Program,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = (
        cluster_sampling_temperature_period)

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs: int = 0

  def register_program(
      self,
      program: code_manipulation.Function,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    signature = _get_signature(scores_per_test)
    print(f'signature:{signature}')
    if signature not in self._clusters:
      print("signature not in self._clusters")
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(program)
    self._num_programs += 1

  def get_prompt(self) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])
    print(f'cluster_scores type: {type(cluster_scores)}') 

    # Convert scores to probabilities using softmax with temperature schedule.
    period = self._cluster_sampling_temperature_period
    temperature = self._cluster_sampling_temperature_init * (
        1 - (self._num_programs % period) / period)
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)
    print(f' len cluster: {len(self._clusters)}')
    print(f'functions_per_prompt: {functions_per_prompt}')

    idx = np.random.choice(
        len(signatures), size=functions_per_prompt, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    print(f'chosen_signatures: {chosen_signatures}')
    implementations = []
    self.scores = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      self.scores.append(cluster.score)

    indices = np.argsort(self.scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    print("##################################") 
    print(f'# impelementations passed in the prompt: {len(implementations)}')

    implementations = copy.deepcopy(implementations)  # We will mutate these.
    print(f'impelementations passed in the prompt: {implementations}')

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      #print(f'Implementation in programs database: {vars(implementation)}')
      #new_function_name = f'{self._function_to_evolve}_v{i}'
      new_function_name = f'agent_v{i}'
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.

      if i >= 1:
        implementation.docstring = (
            f"Improved version of `{self._function_to_evolve}_v{i - 1}`")# with score:{self.scores[i]}.")
      else: 
        implementation.docstring = (f"score of this function is: {self.scores[i]}.")
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      versioned_functions.append(
          code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    '''
    print("####################################")
    print("####################################")
    print(f'prompt:{str(prompt)}')
    print("####################################")
    print("####################################")
    '''
    return str(prompt)


class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function):
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorter programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    return np.random.choice(self._programs)#, p=probabilities)


