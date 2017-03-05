#!/bin/bash
# This script must be run with sudo.
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
wget https://github.com/venky18/venky18.github.io/blob/master/Makefile.config
cd python
for req in $(cat requirements.txt); do sudo -H pip install $req --upgrade; done
cd ..
make all -j4
make test -j4
make runtest -j4
make pycaffe -j4
