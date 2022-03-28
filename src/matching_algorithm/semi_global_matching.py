# Tobit Flatscher - github.com/2b-t (2022)

# @file semi_global_matching.py
# @brief Semi-global matching (SGM) stereo matching algorithm

import abc
from numba import jit
import numpy as np
from scipy.sparse import diags

from .matching_algorithm import MatchingAlgorithm


class SemiGlobalMatching(MatchingAlgorithm):

  @staticmethod
  def match(cost_volume: np.ndarray) -> np.ndarray:
    # Function for matching the best suiting pixels for the disparity image
    #   @param[in] cost_volume: The three-dimensional cost volume to be searched for the best matching pixel (H,W,D)
    #   @return: The two-dimensional disparity image resulting from the best matching pixel inside the cost volume (H,W)

    (_, _, max_disparity) = cost_volume.shape
    f = SemiGlobalMatching._get_f(max_disparity)
    return SemiGlobalMatching._compute_sgm(cost_volume, f)

  def _get_f(D: int, L1: float = 0.025, L2: float = 0.5) -> np.ndarray:
    # Get pairwise cost matrix for semi-global matching
    #   @param[in] D: Maximum disparity, number of possible choices
    #   @param[in] L1: Parameter for setting cost for jumps between two layers of depth
    #   @param[in] L2: Cost for jumping more than one layer of depth
    #   @return: Pairwise_costs of shape (D,D)
    
    return np.full((D, D), L2) + diags([L1 - L2, -L2, L1 - L2], [-1, 0, 1], (D, D)).toarray()

  # For some reason @jit(nopython = True, parallel = True, cache = True) does not work here!
  # See Issue #1: https://github.com/2b-t/stereo-matching/issues/1
  @staticmethod
  @jit
  def _compute_message(cost_volume: np.ndarray, f: np.ndarray) -> np.ndarray:
    # Compute the messages in one particular direction for semi-global matching
    #
    #   @param[in] cost_volume: Cost volume of shape (H,W,D)
    #   @param[in] f: Pairwise costs of shape (D,D)
    #   @return: Messages for all H in positive direction of W with possible options D (H,W,D)
    
    (H,W,D) = cost_volume.shape
    mes = np.zeros((H,W,D))
    # Loop over passive direction
    for y in range(0, H):
      # Loop over forward direction
      for x in range(0, W - 1):
        # Loop over all possible nodes
        for t in range(0, D):

          # Loop over all possible connections
          buffer = np.zeros(D)
          for s in range(0, D):
            # Input messages + unary cost + binary cost
            buffer[s] = mes[y,x,s] + cost_volume[y,x,s] + f[t,s]
          
          # Choose path of least effort
          mes[y, x+1, t] = np.min(buffer)
    
    return mes

  @staticmethod
  def _compute_sgm(cost_volume: np.ndarray, f: np.ndarray) -> np.ndarray:
    # Compute semi-global matching by message passing in four directions
    #   @param[in] cost_volume: Cost volume of shape (H,W,D)
    #   @param[in] f: Pairwise costs of shape (H,W,D,D)
    #   @return: Pixel-wise disparity map of shape (H,W)
    
    # Messages for every single spatial direction and collect in single message
    (H,W,D) = cost_volume.shape
    mes = np.zeros((H,W,D))
    
    # Positive W
    mes += SemiGlobalMatching._compute_message(cost_volume, f)
    
    # Negative W
    mes_buffer = np.zeros((H,W))
    mes_buffer = SemiGlobalMatching._compute_message(np.flip(cost_volume, axis=1), f)
    mes += np.flip(mes_buffer, axis=1)
    
    # Positive H
    mes_buffer = SemiGlobalMatching._compute_message(np.transpose(cost_volume, (1, 0, 2)), f)
    mes += np.transpose(mes_buffer, (1, 0, 2))
    
    # Negative H
    mes_buffer = SemiGlobalMatching._compute_message(np.flip(np.transpose(cost_volume, (1, 0, 2)), axis=1), f)
    mes += np.transpose(np.flip(mes_buffer, axis=1), (1, 0, 2))
    
    # Choose best believe from all messages
    disp_map = np.zeros((H,W))
    for y in range(0, H):
      for x in range(0, W):
        # Minimum argument of unary cost and messages
        disp_map[y,x] = np.argmin(cost_volume[y,x,:] + mes[y,x,:])
    
    return disp_map
