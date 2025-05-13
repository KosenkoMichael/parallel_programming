cmake_minimum_required(VERSION 3.10)
project(cuda_matrix_mul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

add_executable(cuda_matrix_mul main.cu)
