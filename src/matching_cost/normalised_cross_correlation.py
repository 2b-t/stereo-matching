from numba import jit
import numpy as np

from .matching_cost import MatchingCost


class NormalisedCrossCorrelation(MatchingCost):

  @staticmethod
  @jit(nopython = True, parallel = True, cache = True)
  def compute(left_image: np.ndarray, right_image: np.ndarray, max_disparity: int, filter_radius: int) -> np.ndarray:
    # Compute a cost volume with maximum disparity D considering a neighbourhood R with Normalized Cross Correlation (NCC)
    #   @param[in] left_image: The left image to be used for stereo matching (H,W)
    #   @param[in] right_image: The right image to be used for stereo matching (H,W)
    #   @param[in] max_disparity: The maximum disparity to consider
    #   @param[in] filter_radius: The filter radius to be considered for matching
    #   @return: The best matching pixel inside the cost volume according to the pre-defined criterion (H,W,D)
    
    (H,W) = left_image.shape
    cost_volume = np.zeros((max_disparity,H,W))
    
    # Loop over all possible disparities
    for d in range(0, max_disparity):
      # Loop over image
      for y in range(filter_radius, H - filter_radius):
        for x in range(filter_radius, W - filter_radius):
          l_mean = 0
          r_mean = 0
          n = 0
          
          # Loop over window
          for v in range(-filter_radius, filter_radius + 1):
            for u in range(-filter_radius, filter_radius + 1):     
              # Calculate cumulative sum
              l_mean += left_image[y+v, x+u]
              r_mean += right_image[y+v, x+u-d]
              n  += 1
          
          l_mean = l_mean/n
          r_mean = r_mean/n
          
          l_r = 0
          l_var = 0
          r_var = 0
          
          for v in range(-filter_radius, filter_radius + 1):
            for u in range(-filter_radius, filter_radius + 1):     
              # Calculate terms
              l = left_image[y+v, x+u]    - l_mean
              r = right_image[y+v, x+u-d] - r_mean
              
              l_r   += l*r
              l_var += l**2
              r_var += r**2
          
          # Assemble terms
          cost_volume[d,y,x] = -l_r/np.sqrt(l_var*r_var)
    
    return np.transpose(cost_volume, (1, 2, 0))