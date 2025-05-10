cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplicationMPI)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

set(MPI_HOME "C:/Program Files (x86)/Microsoft SDKs/MPI")
set(CMAKE_CXX_STANDARD 17)

find_path(MPI_INCLUDE_PATH mpi.h PATHS "${MPI_HOME}/Include")
find_library(MPI_LIBRARY NAMES msmpi PATHS "${MPI_HOME}/Lib/x64")

add_executable(matrix_mpi src/main.cpp)
target_include_directories(matrix_mpi PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(matrix_mpi PRIVATE ${MPI_LIBRARY})