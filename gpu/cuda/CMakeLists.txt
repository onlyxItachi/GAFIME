cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(gafime_cuda_v1 SHARED
    device.cu
    host.cpp
    tune.cpp
)

set_source_files_properties(host.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(gafime_cuda_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23>
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++23>
)
target_include_directories(gafime_cuda_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
target_link_libraries(gafime_cuda_v1 PRIVATE CUDA::cudart)
