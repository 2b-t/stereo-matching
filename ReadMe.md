# Stereo matching

*Author: Tobit Flatscher (January 2020)

## Overview
This small tool is a **manual implementation of simple stereo-matching**. Two rectified images taken from different views:

Left image             |  Right image
:-------------------------:|:-------------------------:
![Left image](/data/Adirondack_left.png) | ![Right image](/data/Adirondack_right.png)

are combined to a **depth image** by means of two **matching algorithms**, a simple **winner-takes-it-all (WTA)** or more sophisticated **semi-global matching (SGM)** with several **matching costs** (**Sum of Absolute Differences (SAD), Sum of Squared Differences (SAD) or Normalized Cross-Correlation (NCC)**).

![Depth image](/output/Adirondack_NCC_SGM_D70_R3_appX0,92.jpg)

The results are compared to a ground-truth using the accX accuracy measure excluding occluded pixels with a mask.

### Formulas
For the precise details of the involved formulas (matching cost, matching algorithms and accuracy measure) refer to `Theory.pdf`.

### The files
- `/data/...` Directory for the input images (left and right eye)
- `/output/...` Directory for the resulting depth-image output
- `Main.ipynb` The Jupyter notebook
- `stereo_matching.py` The Python implementation of the core functions with Numba and Numpy
- `Theory.pdf` Explanation of the involved formulas

## Launch it
Copy this folder or directly **clone this repository** by typing
```
$ git clone https://github.com/2b-t/SciPy-stereo-sgm.git
```
Make sure Numba, Numpy and Jupyter are installed on your system by typing
```
$ jupyter --version
```
If they are not installed on your system yet, install them - ideally with [Anaconda](https://www.anaconda.com/distribution/) - and launch Jupyter notebook by typing
```
$ jupyter notebook
```
Open the Jupyter notebook and run it by pressing the play-button. Alternatively you can also edit the Python-file `stereo_matching.py` in your editor of choice (e.g. PyCharm) and launch it from there.
