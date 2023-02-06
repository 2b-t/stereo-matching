FROM ubuntu:20.04

WORKDIR /code

ARG DEBIAN_FRONTEND=noninteractive

# General tools
RUN apt-get update \
 && apt-get install -y \
    build-essential \
    cmake \
    git-all \
 && rm -rf /var/lib/apt/lists/*

# Python3 and libraries
RUN apt-get update \
 && apt-get install -y \
    python3 \
    python3-scipy \
    python3-skimage \
    python3-numpy \
    python3-numba \
    python3-notebook \
    python3-matplotlib \
    python3-parameterized \
 && rm -rf /var/lib/apt/lists/*

# For visualisation of Jupyter notebooks
RUN apt-get update \
 && apt-get install -y \
    firefox \
 && rm -rf /var/lib/apt/lists/*

# For documentation only
#RUN apt-get update \
# && apt-get install -y \
#    texlive-full \
#    texstudio \
# && rm -rf /var/lib/apt/lists/*
 
 ARG DEBIAN_FRONTEND=dialog

