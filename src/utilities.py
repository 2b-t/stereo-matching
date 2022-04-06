# Tobit Flatscher - github.com/2b-t (2022)

# @file utilities.py
# @brief Different utilities for AccX accuracy measure and file input and output

import numpy as np
import os

from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.color import rgb2gray


class AccX:
  # Class for the AccX accuracy measure

  @staticmethod
  def compute(prediction_image: np.ndarray, groundtruth_image: np.ndarray, mask_image: np.ndarray = None, threshold_disparity: int = 3) -> float:
    # Compute the accX accuracy measure [0..1]
    #   @param[in] prediction_image: The stereo image as reconstructed by an algorithm
    #   @param[in] groundtruth_image: The ground truth stereo image
    #   @param[in] mask_image: The mask for excluding invalid pixels such as occluded areas
    #   @param[in] threshold_disparity: Threshold disparity measure (X)
    #   @return The accX measure of the reconstructed stereo image
    
    if (prediction_image.shape != groundtruth_image.shape):
      raise ValueError("Dimensions of guess (" + prediction_image.shape + ") and groundtruth (" + groundtruth_image.shape + ") do not match.")
    
    if (mask_image is None):
      mask_image = np.ones(prediction_image.shape)
    
    number_of_pixels = max(np.sum(mask_image), 1) # Catch error if no pixels selected
    
    weighted_image = mask_image*(np.absolute(prediction_image - groundtruth_image) <= threshold_disparity)
    return 1/number_of_pixels*np.sum(weighted_image)


class IO:
  # Class for input output tools

  @staticmethod
  def import_image(file_name: str) -> np.ndarray:
    # Import image and convert it to useable grey-scale image
    # @param[in] file_name: The file name of the file to be imported
    # @return The parsed image as a numpy array
    img = imread(file_name)
    return rgb2gray(img)
  
  @staticmethod
  def export_image(image: np.ndarray, directory: str, name: str, matching_cost: str, matching_algorithm: str, 
                   max_disparity: int, filter_radius: int, accx = None) -> str:
    # Export image to disk with an approriate file name
    #   @param[in] export_image: The image data that has to be exported as numpy array
    #   @param[in] directory: Sub-directory where the file should be saved
    #   @param[in] name: Scenario name
    #   @param[in] matching_cost: The matching cost used (e.g. SSD, SAD, NCC)
    #   @param[in] matching_algorithm: The measure used for matching point (e.g. WTA, SGM)
    #   @param[in] max_disparity: Maximum disparity
    #   @param[in] filter_radius: Filter radius
    #   @param[in] accx: accX measure for evaluation (if available)
    #   @return: The resulting file name

    if directory is None:
      directory = ""
    elif not os.path.isdir(directory):
      os.mkdir(directory)

    if name is None:
      name = ""

    path = os.path.join(directory, name)

    file_name = str(path) + "_" + matching_cost + "_" + matching_algorithm + "_D" + IO._str_comma(max_disparity) + "_R" + IO._str_comma(filter_radius)

    if accx is not None:
      file_name += "_accX" + IO._str_comma(accx)

    file_name = file_name + ".jpg"
    imsave(file_name, img_as_ubyte(image), quality = 100)
    return file_name

  @staticmethod
  def _str_comma(number: float, number_of_decimals: int = 2) -> str:
    # Create a string from a number and replace all dots by a comma
    #   @param[in] number: A number that should be converted to a string
    #   @param[in] number_of_decimals: Number of decimals to be kept
    #   @return: A string of the number with 2 decimals where all dots are replaced by commas

    return str(round(number, number_of_decimals)).replace('.',',')
        
  @staticmethod
  def normalise_image(image: np.ndarray, groundtruth_image: np.ndarray = None) -> np.ndarray:
    # Normalise image with groundtruth or itself to floating number points in interval 0..1
    #   @param[in] image: Non-normalised image
    #   @param[in] groundtruth_image: Ground-truth
    #   @return: Image normalised with ground truth or maximimum distance
    
    normalised_image = image
    
    if groundtruth_image is not None:
      if (np.max(groundtruth_image) <= 0):
        raise ValueError("Maximum value in groundtruth image must be greater than 0.")
      normalised_image = image/np.max(groundtruth_image)
    
    if (np.max(image) <= 0):
      raise ValueError("Maximum value in image must be greater than 0.")

    return normalised_image/np.max(normalised_image)
  