export PATH=$PATH:/usr/local/cuda-11.4/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64/

rm -rf CMakeFiles/ CMakeCache.txt

export CC=gcc-10
export CXX=g++-10

cmake .
