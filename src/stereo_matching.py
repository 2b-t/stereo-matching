# Tobit Flatscher - github.com/2b-t (2022)

# @file stereo_matching.py
# @brief Interface class for setting up stereo matching

from enum import Enum
import numpy as np

from matching_algorithm.matching_algorithm import MatchingAlgorithm
from matching_cost.matching_cost import MatchingCost


class StereoMatching:
  # Recreate the depth image from two images with a given maximum disparity to consider and given filter radius

  def __init__(self, left_image: np.ndarray, right_image: np.ndarray,
                     matching_cost: MatchingCost, 
                     matching_algorithm: MatchingAlgorithm, 
                     max_disparity: int = 60, filter_radius: int = 3):
    # Class constructor
    #   @param[in] left_image: The left stereo image (H,W)
    #   @param[in] right_image: The right stereo image (H,W)
    #   @param[in] matching_cost: The class implementing the matching cost
    #   @param[in] matching_algorithm: The class implementing the matching algorithm
    #   @param[in] max_disparity: The maximum disparity to consider
    #   @param[in] filter_radius: The radius of the filter

    if (left_image.ndim != 2):
      raise ValueError("The left image has to be a grey-scale image with a single channel as its last dimension.")
    if (right_image.ndim != 2):
      raise ValueError("The right image has to be a grey-scale image with a single channel as its last dimension.")
    if (left_image.shape != right_image.shape):
      raise ValueError("Dimensions of left (" + left_image.shape + ") and right image (" + right_image.shape + ") do not match.")
    if (max_disparity <= 0):
      raise ValueError("Maximum disparity (" + max_disparity + ") has to be greater than zero.")
    if (filter_radius <= 0):
      raise ValueError("Radius (" + filter_radius + ") has to be greater than zero.")

    # Convert images to gray-scale
    self._left_image = left_image
    self._right_image = right_image

    self._max_disparity = max_disparity
    self._filter_radius = filter_radius
    self._matching_cost = matching_cost
    self._matching_algorithm = matching_algorithm
    self._cost_volume = None
    self._result = None
    return

  def compute(self) -> None:
    # Compute the cost volume according to given matching cost and match according matching algorithm

    self._cost_volume = self._matching_cost.compute(self._left_image, self._right_image, self._max_disparity, self._filter_radius)
    self._result = self._matching_algorithm.match(self._cost_volume)
    return
  
  def result(self) -> np.ndarray:
    # Export image to disk with an approriate file name
    #   @return: The generated result image or None if the image has not been generated yet

    return self._result
