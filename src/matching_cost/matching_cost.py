# Tobit Flatscher - github.com/2b-t (2022)

# @file matching_cost.py
# @brief Base class for stereo matching costs

import abc
import numpy as np


class MatchingCost(abc.ABC):
  # Base class for stereo matching costs for calculating a cost volume

  @staticmethod
  @abc.abstractmethod
  def compute(self, left_image: np.ndarray, right_image: np.ndarray, max_disparity: int, filter_radius: int) -> np.ndarray:
    # Function for calculating the cost volume
    #   @param[in] left_image: The left image to be used for stereo matching (H,W)
    #   @param[in] right_image: The right image to be used for stereo matching (H,W)
    #   @param[in] max_disparity: The maximum disparity to consider
    #   @param[in] filter_radius: The filter radius to be considered for matching
    #   @return: The best matching pixel inside the cost volume according to the pre-defined criterion (H,W,D)

    pass
