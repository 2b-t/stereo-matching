FROM ubuntu:20.04

WORKDIR /code

ARG DEBIAN_FRONTEND=noninteractive

# Install general tools: Git, Cmake, GCC
RUN apt-get -y update && apt-get -y install \
    build-essential \
    git-all \
    cmake

# Install Python3, Scipy, Numpy, Numba and unittest
RUN apt-get -y update && apt-get -y install \
    python3 \
    python3-scipy \
    python3-skimage \
    python3-numpy \
    python3-numba \
    python3-notebook \
    python3-matplotlib

# For Visualisation of Jupyter notebooks
RUN apt-get -y update && apt-get -y install \
    firefox

# For documentation
RUN apt-get -y update && apt-get -y install \
    texlive-full \
    texstudio


