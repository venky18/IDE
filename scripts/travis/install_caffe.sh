#!/bin/bash
# This script must be run with sudo.
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y libatlas-base-dev 
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
# (Python general)
sudo apt-get install -y python-pip
# (Python 2.7 development files)
sudo apt-get install -y python-dev
sudo apt-get install -y python-numpy python-scipy
# (OpenCV 2.4)
sudo apt-get install -y libopencv-dev
cd $HOME/tools
git clone https://github.com/BVLC/caffe.git
cd caffe
#configure cmake file

LINE () {
  echo "$@" >> Makefile.config
}

cp Makefile.config.example Makefile.config

LINE "BLAS := atlas"
LINE "WITH_PYTHON_LAYER := 1"
LINE "CPU_ONLY := 1"
LINE "INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial"
LINE "LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial"
LINE "BUILD_DIR := build"
LINE "DISTRIBUTE_DIR := distribute"

'''

BLAS := atlas

PYTHON_INCLUDE := /usr/include/python2.7 \
    /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_LIB := /usr/lib
WITH_PYTHON_LAYER := 1

INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial


BUILD_DIR := build
DISTRIBUTE_DIR := distribute
Q ?= @

'''