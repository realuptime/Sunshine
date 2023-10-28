export PATH=$PATH:/usr/local/cuda-11.4/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64/

export CC=gcc-10
export CXX=g++-10

rm -rf CMakeFiles/ CMakeCache.txt; cmake .
#make

#sudo setcap cap_sys_admin+ep sunshine-0.21.0.dirty

