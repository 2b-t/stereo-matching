#!/usr/bin/env python3
# Tobit Flatscher - github.com/2b-t (2022)

# @file main.py
# @brief Command line interface for stereo matching

import argparse
import matplotlib.pyplot as plt
import numpy as np

from matching_algorithm.matching_algorithm import MatchingAlgorithm
from matching_algorithm.semi_global_matching import SemiGlobalMatching
from matching_algorithm.winner_takes_it_all import WinnerTakesItAll

from matching_cost.matching_cost import MatchingCost
from matching_cost.normalised_cross_correlation import NormalisedCrossCorrelation
from matching_cost.sum_of_absolute_differences import SumOfAbsoluteDifferences
from matching_cost.sum_of_squared_differences import SumOfSquaredDifferences

from stereo_matching import StereoMatching
from utilities import AccX, IO


def main(left_image_path: str, right_image_path: str, 
         matching_algorithm_name: str, matching_cost_name: str, 
         max_disparity: int, filter_radius: int, 
         groundtruth_image_path: str, mask_image_path: str, accx_threshold: int,
         output_path: str = None, output_name: str = "unknown", is_plot: bool = True) -> None:
  # Imports images for stereo matching, performs stereo matching, plots results and outputs them to a file
  #   @param[in] left_image_path:          Path to the image for the left eye
  #   @param[in] right_image_path:         Path to the image for the right eye
  #   @param[in] matching_algorithm_name:  Name of the matching algorithm
  #   @param[in] matching_cost_name:       Name of the matching cost type
  #   @param[in] max_disparity:            Maximum disparity to consider
  #   @param[in] filter_radius:            Filter radius to be considered for cost volume
  #   @param[in] groundtruth_image_path:   Path to the ground truth image
  #   @param[in] mask_image_path:          Path to the mask for excluding pixels from the AccX accuracy measure
  #   @param[in] accx_threshold:           Mismatch in disparity to accept for AccX accuracy measure
  #   @param[in] output_path:              Location of the output path, if None no output is generated
  #   @param[in] output_name:              Name of the scenario for pre-pending the output file
  #   @param[in] is_plot:                  Flag for turning plot of results on and off
  
  # Load input images
  left_image = IO.import_image(left_image_path)
  right_image = IO.import_image(right_image_path)

  # Load ground truth images
  groundtruth_image = None
  mask_image = None
  try:
    groundtruth_image = IO.import_image(groundtruth_image_path)
    mask_image = IO.import_image(mask_image_path)
  except:
    pass

  # Plot input images
  if is_plot is True:
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1), plt.imshow(left_image, cmap='gray'), plt.title('Left')
    plt.subplot(1,2,2), plt.imshow(right_image, cmap='gray'), plt.title('Right')
    plt.tight_layout()

  # Set-up algorithm
  matching_algorithm = None
  if matching_algorithm_name == "SGM":
    matching_algorithm = SemiGlobalMatching
  elif matching_algorithm_name == "WTA":
    matching_algorithm = WinnerTakesItAll
  else:
    raise ValueError("Matching algorithm '" + matching_algorithm_name + "' not recognised!")
  
  matching_cost = None
  if matching_cost_name == "NCC":
    matching_cost = NormalisedCrossCorrelation
  elif matching_cost_name == "SAD":
    matching_cost = SumOfAbsoluteDifferences
  elif matching_cost_name == "SSD":
    matching_cost = SumOfSquaredDifferences
  else:
    raise ValueError("Matching cost '" + matching_cost_name + "' not recognised!")

  # Perform stereo matching
  sm = StereoMatching(left_image, right_image, matching_cost, matching_algorithm, max_disparity, filter_radius)
  print("Performing stereo matching...")
  sm.compute()
  print("Stereo matching completed.")
  res_image = sm.result()

  # Compute accuracy
  try:
    accx = AccX.compute(res_image, groundtruth_image, mask_image, accx_threshold)
    print("AccX accuracy measure for threshold " + str(accx_threshold) + ": " + str(accx))
  except:
    accx = None

  # Plot result
  if is_plot is True:
    plt.figure()
    plt.imshow(res_image, cmap='gray')
    plt.show()
  
  # Output to file
  if output_path is not None:
    result_file_path = IO.export_image(IO.normalise_image(res_image, groundtruth_image), 
                                       output_path, output_name, matching_cost_name, matching_algorithm_name, 
                                       max_disparity, filter_radius, accx_threshold)
    print("Exported result to file '" + result_file_path + "'.")
  return


if __name__== "__main__":
  # Parse input arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-l", "--left", type=str, 
                      help="Path to left image")
  parser.add_argument("-r", "--right", type=str, 
                      help="Path to right image")
  parser.add_argument("-a", "--algorithm", type=str, choices=["SGM", "WTA"],
                      help="Matching cost algorithm", default = "WTA")
  parser.add_argument("-c", "--cost", type=str, choices=["NCC", "SAD", "SSD"],
                      help="Matching cost type", default = "SAD")
  parser.add_argument("-D", "--disparity", type=int, 
                      help="Maximum disparity", default = 60)
  parser.add_argument("-R", "--radius", type=int, 
                      help="Filter radius", default = 3)
  parser.add_argument("-o", "--output", type=str, 
                      help="Output directory, by default no output", default = None)
  parser.add_argument("-n", "--name", type=str, 
                      help="Output file name", default = "unknown")
  parser.add_argument("-p", "--no-plot", action='store_true', 
                      help="Flag for de-activating plotting")
  parser.add_argument("-g", "--groundtruth", type=str, 
                      help="Path to groundtruth image", default = None)
  parser.add_argument("-m", "--mask", type=str, 
                      help="Path to mask image for AccX accuracy measure", default = None)
  parser.add_argument("-X", "--accx", type=int, 
                      help="AccX accuracy measure threshold", default = 60)
  args = parser.parse_args()

  main(args.left, args.right, args.algorithm, args.cost, args.disparity, args.radius, 
       args.groundtruth, args.mask, args.accx, 
       args.output, args.name, not args.no_plot)
