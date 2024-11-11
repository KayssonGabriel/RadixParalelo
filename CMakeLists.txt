cmake_minimum_required(VERSION 3.29)
project(RadixSortCUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(RadixSortCUDA main.cu)

set_target_properties(RadixSortCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
