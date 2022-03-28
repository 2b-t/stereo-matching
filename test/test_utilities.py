# Tobit Flatscher - github.com/2b-t (2022)

# @file utilities_test.py
# @brief Different testing routines for utility functions for accuracy calculation and file import and export

import numpy as np
from parameterized import parameterized
import unittest

from src.utilities import AccX


class TestAccX(unittest.TestCase):
  @parameterized.expand([
    ["disparity = 1", 1],
    ["disparity = 2", 2],
    ["disparity = 3", 3]
  ])
  def test_same_image(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if two identical images result in an accuracy measure of unity
    #   @param[in] name: The name of the parameterised test
    #   @param[in] name: The threshold disparity for the accuracy measure

    shape = (10,20)
    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(shape)
    prediction_image = mag*np.ones(groundtruth_image.shape)
    mask_image = np.ones(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 1.0, places=7)
    return

  @parameterized.expand([
    ["disparity = 1", 1],
    ["disparity = 2", 2],
    ["disparity = 3", 3]
  ])
  def test_inverse_image(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if two inverse images result in an accuracy measure of zero
    #   @param[in] name: The name of the parameterised test
    #   @param[in] name: The threshold disparity for the accuracy measure

    shape = (10,20)
    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(shape)
    prediction_image = np.zeros(groundtruth_image.shape)
    mask_image = np.ones(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 0.0, places=7)
    return
  
if __name__ == '__main__':
  unittest.main()