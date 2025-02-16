""" Finds the result of adding 2 to the input

improve priority_vX function to produce priority_v(X+1) to make a better way of adding 2 to the input 
make only small changes to the code 
make the code as short as possible 
"""

import os 
import sys 

import numpy as np 

file_path = os.path.abspath('/Users/ssarch/Documents/emerge/deepmind-implementation/funsearch/funsearch')

sys.path.append(os.path.abspath(os.path.join(file_path, '..')))

import funsearch


@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of an `n`-dimensional cap set."""
  return n-2


def solve(n: int) -> np.ndarray:
  """Returns a large cap set in `n` dimensions."""

@funsearch.evolve
def priority( n: int) -> float:
  """
  returns the value of the input after adding 2 to it. 
  """
  return n+2

