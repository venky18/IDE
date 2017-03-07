#!/bin/bash
# This script must be run with sudo.
ls
echo "look here"
source ./scripts/travis/install_dep.sh

cd $HOME/tools
git clone https://github.com/BVLC/caffe.git
cd caffe
# #configure cmake file
source ./scripts/travis/configure-make.sh
cd python
for req in $(cat requirements.txt); do sudo -H pip install $req --upgrade; done
cd ..
# # wget https://raw.githubusercontent.com/venky18/venky18.github.io/master/Makefile.config
# make all -j4
# make test -j4
# make runtest -j4
# make pycaffe -j4
make --jobs $NUM_THREADS all test pycaffe warn
make runtest
make pytest
make pycaffe 
export PYTHONPATH=/path/to/caffe-master/python:$PYTHONPATH
# git clone https://github.com/venky18/IDE/tree/travis
# cd IDE/scripts/travis
# source ./build.sh
# source ./test.sh