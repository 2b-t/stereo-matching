# Tobit Flatscher - github.com/2b-t (2022)

# @file sum_of_absolute_differences.py
# @brief Sum of absolute differences (SAD) stereo matching cost

from numba import jit
import numpy as np

from .matching_cost import MatchingCost


class SumOfAbsoluteDifferences(MatchingCost):

  @staticmethod
  @jit(nopython = True, parallel = True, cache = True)
  def compute(left_image: np.ndarray, right_image: np.ndarray, max_disparity: int, filter_radius: int) -> np.ndarray:
    # Compute a cost volume with maximum disparity D considering a neighbourhood R with Sum of Absolute Differences (SAD)
    #   @param[in] left_image: The left image to be used for stereo matching (H,W)
    #   @param[in] right_image: The right image to be used for stereo matching (H,W)
    #   @param[in] max_disparity: The maximum disparity to consider
    #   @param[in] filter_radius: The filter radius to be considered for matching
    #   @return: The best matching pixel inside the cost volume according to the pre-defined criterion (H,W,D)
    
    (H,W) = left_image.shape
    cost_volume = np.zeros((H,W,max_disparity))
    
    # Loop over internal image
    for y in range(filter_radius, H - filter_radius):
      for x in range(filter_radius, W - filter_radius):  
        # Loop over window
        for v in range(-filter_radius, filter_radius + 1):
          for u in range(-filter_radius, filter_radius + 1):
            # Loop over all possible disparities
            for d in range(0, max_disparity):
              cost_volume[y,x,d] += np.absolute(left_image[y+v, x+u] - right_image[y+v, x+u-d])
        
    return cost_volume