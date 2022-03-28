import abc
import numpy as np

from .matching_algorithm import MatchingAlgorithm


class WinnerTakesItAll(MatchingAlgorithm):

  @staticmethod
  def match(cost_volume: np.ndarray) -> np.ndarray:
    # Function for matching the best suiting pixels for the disparity image
    #   @param[in] cost_volume: The three-dimensional cost volume to be searched for the best matching pixel (H,W,D)
    #   @return: The two-dimensional disparity image resulting from the best matching pixel inside the cost volume (H,W)
    
    return np.argmin(cost_volume, axis=2)