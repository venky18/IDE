# sudo apt-get install -y build-essential cmake git pkg-config
# sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
# sudo apt-get install -y libatlas-base-dev 
# sudo apt-get install -y --no-install-recommends libboost-all-dev
# sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
# # (Python general)
# sudo apt-get install -y python-pip
# # (Python 2.7 development files)
# sudo apt-get install -y python-dev
# sudo apt-get install -y python-numpy python-scipy
# # glog
# wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
# tar zxvf glog-0.3.3.tar.gz
# cd glog-0.3.3
# ./configure
# make && make install
# # gflags
# wget https://github.com/schuhschuh/gflags/archive/master.zip
# unzip master.zip
# cd gflags-master
# mkdir build && cd build
# export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
# make && make install
# # lmdb
# git clone https://github.com/LMDB/lmdb
# cd lmdb/libraries/liblmdb
# make && make install
# # (OpenCV 2.4)
# sudo apt-get install -y libopencv-dev#!/bin/bash
# install dependencies
# (this script must be run as root)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y libblas-dev  libatlas-dev libatlas-base-dev 
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
apt-get install -y --no-install-recommends \
  build-essential \
  libboost-filesystem-dev \
  libboost-python-dev \
  libboost-system-dev \
  libboost-thread-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libhdf5-serial-dev \
  libopenblas-dev \
  python-virtualenv \
  wget

wget https://github.com/schuhschuh/gflags/archive/master.zip
unzip master.zip
cd gflags-master
mkdir build && cd build
export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
make && make install
# lmdb
git clone https://github.com/LMDB/lmdb
cd lmdb/libraries/liblmdb
make && make install

if $WITH_CMAKE ; then
  apt-get install -y --no-install-recommends cmake
fi

if ! $WITH_PYTHON3 ; then
  # Python2
  apt-get install -y --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-protobuf \
    python-skimage
else
  # Python3
  apt-get install -y --no-install-recommends \
    python3-dev \
    python3-numpy \
    python3-skimage

  # build Protobuf3 since it's needed for Python3
  PROTOBUF3_DIR=~/protobuf3
  pushd .
  if [ -d "$PROTOBUF3_DIR" ] && [ -e "$PROTOBUF3_DIR/src/protoc" ]; then
    echo "Using cached protobuf3 build ..."
    cd $PROTOBUF3_DIR
  else
    echo "Building protobuf3 from source ..."
    rm -rf $PROTOBUF3_DIR
    mkdir $PROTOBUF3_DIR

    # install some more dependencies required to build protobuf3
    apt-get install -y --no-install-recommends \
      curl \
      dh-autoreconf \
      unzip

    wget https://github.com/google/protobuf/archive/3.0.x.tar.gz -O protobuf3.tar.gz
    tar -xzf protobuf3.tar.gz -C $PROTOBUF3_DIR --strip 1
    rm protobuf3.tar.gz
    cd $PROTOBUF3_DIR
    ./autogen.sh
    ./configure --prefix=/usr
    make --jobs=$NUM_THREADS
  fi
  make install
  popd
fi

if $WITH_IO ; then
  apt-get install -y --no-install-recommends \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libsnappy-dev
fi