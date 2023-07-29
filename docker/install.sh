#!/bin/bash

apt update -qq
apt install pkg-config -y
apt-get install --no-install-recommends\
    ninja-build \
    doxygen \
    autoconf \
    automake \
    cmake \
    g++ \
    gcc \
    pkg-config \
    make \
    nasm \
    yasm -y


# install vmaf
export PATH="$PATH:$HOME/.local/bin"
wget https://github.com/Netflix/vmaf/archive/v2.3.1.tar.gz
tar -xzf  v2.3.1.tar.gz
cd vmaf-2.3.1/libvmaf/
meson build --buildtype release
ninja -vC build
ninja -vC build test
ninja -vC build install

#install ffmpeg
wget https://ffmpeg.org/releases/ffmpeg-6.0.tar.bz2 && tar xjf ffmpeg-6.0.tar.bz2 
cd ffmpeg-6.0
./configure --enable-libvmaf --ld="g++"
make
make install