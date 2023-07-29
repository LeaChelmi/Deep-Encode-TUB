#!/bin/bash

wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz

wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5

md5sum -c ffmpeg-git-amd64-static.tar.xz.md5

tar xvf ffmpeg-git-amd64-static.tar.xz

