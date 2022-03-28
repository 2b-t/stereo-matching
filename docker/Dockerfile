FROM ubuntu:20.04

WORKDIR /code

ARG DEBIAN_FRONTEND=noninteractive

# General tools
RUN apt-get -y update && apt-get -y install \
    build-essential \
    git-all \
    cmake

# Python3 and libraries
RUN apt-get -y update && apt-get -y install \
    python3 \
    python3-scipy \
    python3-skimage \
    python3-numpy \
    python3-numba \
    python3-notebook \
    python3-matplotlib \
    python3-parameterized

# For visualisation of Jupyter notebooks
RUN apt-get -y update && apt-get -y install \
    firefox

# For documentation
RUN apt-get -y update && apt-get -y install \
    texlive-full \
    texstudio

