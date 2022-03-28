import abc
import numpy as np


class MatchingAlgorithm(abc.ABC):
  # Base class for stereo matching algorithms which finds the best matching pixel

  @staticmethod
  @abc.abstractmethod
  def match(cost_volume: np.ndarray) -> np.ndarray:
    # Function for matching the best suiting pixels for the disparity image
    #   @param[in] cost_volume: The three-dimensional cost volume to be searched for the best matching pixel (H,W,D)
    #   @return: The two-dimensional disparity image resulting from the best matching pixel inside the cost volume (H,W)
    if cost_volume.ndim == 3:
      raise ValueError("Cost volume (" + cost_volume.shape + ") must be three-dimensional!")
    pass