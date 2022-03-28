# Stereo matching

Author: [Tobit Flatscher](https://github.com/2b-t) (January 2020)

## 0. Overview
This small tool is a **manual implementation of simple stereo-matching** in Python 3. Two rectified images taken from different views:

Left image             |  Right image
:-------------------------:|:-------------------------:
![Left image](/data/Adirondack_left.png) | ![Right image](/data/Adirondack_right.png)

are combined to a **depth image** by means of two **matching algorithms**:

- a simple **winner-takes-it-all (WTA)** or 
- a more sophisticated **semi-global matching (SGM)**

with several **matching costs**:

- **Sum of Absolute Differences (SAD)**,
- **Sum of Squared Differences (SSD)** or
- **Normalized Cross-Correlation (NCC)**.

![Depth image](/output/Adirondack_NCC_SGM_D70_R3_accX0,92.jpg)

The results are compared to a ground-truth using the accX accuracy measure excluding occluded pixels with a mask.

For the precise details of the involved formulas (matching cost, matching algorithms and accuracy measure) refer to [`doc/Theory.pdf`](./doc/Theory.pdf).

The repository is structured as follows:

- [`data/`](./data/) Directory for the input images (left and right eye)
- [`doc/Theory.pdf`](./doc/Theory.pdf) Explanation of the involved formulas
- [`docker/`](./docker/) Contains a Docker container as well as a Docker-Compose configuration file
- [`output/`](./output/) Directory for the resulting depth-image output
- [`src/Main.ipynb`](./src/Main.ipynb) The Jupyter notebook that allows a convenient access to the underlying Python functions
- [`src/stereo_matching.py`](./src/stereo_matching.py) The Python 3 implementation of the core functions with Scipy, Scimage, Numba, Numpy and Matplotlib

## 1. Download it
Either download and copy this folder manually or directly **clone this repository** by typing
```
$ git clone https://github.com/2b-t/stereo-matching.git
```
## 2. Launch it

Now you have two options for launching the code. Either you can install all libraries on your system and launch the code there or you can use the Docker container located in [`/docker/`](./docker/).

### 2.1 On your system

For launching the code directly on your system make sure Numba, Numpy and Jupyter are installed on your system by typing

```
$ jupyter --version
```
If they are not installed on your system yet, install them - ideally with [Anaconda](https://www.anaconda.com/distribution/) - and launch Jupyter notebook by typing
```
$ jupyter notebook
```
Browse and open the Jupyter notebook [`src/Main.ipynb`](./src/Main.ipynb) and run it by pressing the play-button. Alternatively you can also edit the Python-file [`src/stereo_matching.py`](./src/stereo_matching.py) in your editor of choice (e.g. Visual Studio Code) and launch it from there.

### 2.2 Run from Docker

This is discussed in detail in the document [`/doc/Docker.md`](./doc/Docker.md).
