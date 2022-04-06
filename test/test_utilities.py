# Tobit Flatscher - github.com/2b-t (2022)

# @file utilities_test.py
# @brief Different testing routines for utility functions for accuracy calculation and file import and export

import numpy as np
from parameterized import parameterized
from typing import Tuple
import unittest

from src.utilities import AccX, IO


class TestAccX(unittest.TestCase):
  _shape = (10,20)
  _disparities = [ ["disparity = 1", 1],
                   ["disparity = 2", 2],
                   ["disparity = 3", 3]
                 ]

  @parameterized.expand(_disparities)
  def test_same_image(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if two identical images result in an accuracy measure of unity
    #   @param[in] name: The name of the parameterised test
    #   @param[in] threshold_disparity: The threshold disparity for the accuracy measure

    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(self._shape)
    prediction_image = mag*np.ones(groundtruth_image.shape)
    mask_image = np.ones(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 1.0, places=7)
    return

  @parameterized.expand(_disparities)
  def test_slightly_shifted_image(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if an image and its slightly shifted counterpart result in an accuracy measure of unity
    #   @param[in] name: The name of the parameterised test
    #   @param[in] threshold_disparity: The threshold disparity for the accuracy measure

    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(self._shape)
    prediction_image = (mag+threshold_disparity-1)*np.ones(groundtruth_image.shape)
    mask_image = np.ones(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 1.0, places=7)
    return
  
  @parameterized.expand(_disparities)
  def test_no_mask(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if two identical images with no given mask result in an accuracy measure of unity
    #   @param[in] name: The name of the parameterised test
    #   @param[in] threshold_disparity: The threshold disparity for the accuracy measure

    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(self._shape)
    prediction_image = mag*np.ones(groundtruth_image.shape)
    mask_image = None
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 1.0, places=7)
    return

  @parameterized.expand(_disparities)
  def test_inverse_image(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if two inverse images result in an accuracy measure of zero
    #   @param[in] name: The name of the parameterised test
    #   @param[in] threshold_disparity: The threshold disparity for the accuracy measure

    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(self._shape)
    prediction_image = np.zeros(groundtruth_image.shape)
    mask_image = np.ones(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 0.0, places=7)
    return

  @parameterized.expand(_disparities)
  def test_significantly_shifted_image(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if an image and its significantly shifted counterpart result in an accuracy measure of zero
    #   @param[in] name: The name of the parameterised test
    #   @param[in] threshold_disparity: The threshold disparity for the accuracy measure

    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(self._shape)
    prediction_image = (mag+threshold_disparity+1)*np.ones(groundtruth_image.shape)
    mask_image = np.ones(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 0.0, places=7)
    return

  @parameterized.expand(_disparities)
  def test_zero_mask(self, name: str, threshold_disparity: int) -> None:
    # Parameterised unit test for testing if two equal images with a mask of zero results in an accuracy measure of zero
    #   @param[in] name: The name of the parameterised test
    #   @param[in] threshold_disparity: The threshold disparity for the accuracy measure

    mag = threshold_disparity*10
    groundtruth_image = mag*np.ones(self._shape)
    prediction_image = groundtruth_image
    mask_image = np.zeros(groundtruth_image.shape)
    accx = AccX.compute(prediction_image, groundtruth_image, mask_image, threshold_disparity)
    self.assertAlmostEqual(accx, 0.0, places=7)
    return


class TestIO(unittest.TestCase):
  _resolutions = [ ["resolution = (10, 20)", (10, 20)],
                   ["resolution = (30,  4)", (30,  4)],
                   ["resolution = (65, 24)", (65, 24)]
                 ]
  def test_import_image(self) -> None:
    # TODO(tobit): Implement

    pass
  
  def test_export_image(self) -> None:
    # TODO(tobit): Implement

    pass

  def test_str_comma(self) -> None:
    # Function for testing conversion of numbers to comma-separated numbers

    self.assertEqual(IO._str_comma(10, 2), "10")
    self.assertEqual(IO._str_comma(9.3, 2), "9,3")
    self.assertEqual(IO._str_comma(1.234, 2), "1,23")
    return
  
  @parameterized.expand(_resolutions)
  def test_normalise_positive_image_no_groundtruth(self, name: str, shape: Tuple[int, int]) -> None:
    # Function for testing normalising a positive image with a no ground-truth should result in a positive image
    #   @param[in] name: The name of the parameterised test
    #   @param[in] shape: The image resolution to be considered for the test
    
    mag = 13
    image = mag*np.ones(shape)
    groundtruth_image = None
    result = IO.normalise_image(image, groundtruth_image)
    self.assertGreaterEqual(np.min(result), 0.0)
    self.assertLessEqual(np.max(result), 1.0)
    return

  @parameterized.expand(_resolutions)
  def test_normalise_positive_image_positive_groundtruth(self, name: str, shape: Tuple[int, int]) -> None:
    # Function for testing normalising a regular image with a regular ground-truth should result in a positive image
    #   @param[in] name: The name of the parameterised test
    #   @param[in] shape: The image resolution to be considered for the test

    mag = 13
    image = mag*np.ones(shape)
    groundtruth_image = 2*image
    result = IO.normalise_image(image, groundtruth_image)
    self.assertGreaterEqual(np.min(result), 0.0)
    self.assertLessEqual(np.max(result), 1.0)
    return

  @parameterized.expand(_resolutions)
  def test_normalise_negative_image_positive_groundtruth(self, name: str, shape: Tuple[int, int]) -> None:
    # Function for testing normalising a negative image which should result in a ValueError
    #   @param[in] name: The name of the parameterised test
    #   @param[in] shape: The image resolution to be considered for the test

    mag = 13
    groundtruth_image = mag*np.ones(shape)
    image = -2*groundtruth_image
    self.assertRaises(ValueError, IO.normalise_image, image, groundtruth_image)
    return
  
  @parameterized.expand(_resolutions)
  def test_normalise_positive_image_negative_groundtruth(self, name: str, shape: Tuple[int, int]) -> None:
    # Function for testing normalising a negative ground-truth which should result in a ValueError
    #   @param[in] name: The name of the parameterised test
    #   @param[in] shape: The image resolution to be considered for the test

    mag = 13
    image = mag*np.ones(shape)
    groundtruth_image = -2*image
    self.assertRaises(ValueError, IO.normalise_image, image, groundtruth_image)
    return


if __name__ == '__main__':
  unittest.main()