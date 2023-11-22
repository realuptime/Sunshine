CUDA_VERSION=11.4

export PATH=$PATH:/usr/local/cuda-$CUDA_VERSION/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64/

rm -rf CMakeFiles/ CMakeCache.txt

# CUDA Workaround
# $ sudo update-alternatives --set gcc "/usr/bin/gcc-10"
# $ sudo update-alternatives --set gcc "/usr/bin/gcc-10"

export CC=gcc-10
export CXX=g++-10

cmake .

#sudo setcap cap_sys_admin+ep sunshine-0.21.0.dirty

