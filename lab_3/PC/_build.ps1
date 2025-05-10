set CC=mpicc
set CXX=mpic++

mkdir build
cd build
cmake ..
cmake --build .

exit