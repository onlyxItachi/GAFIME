cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(gafime_cuda_v1 SHARED
    kernels.cu
    launcher.cu
)

if(NOT DEFINED GAFIME_CUDA_ARCHITECTURES)
    set(GAFIME_CUDA_ARCHITECTURES "89-real;89-virtual")
endif()
set_target_properties(gafime_cuda_v1 PROPERTIES
    CUDA_ARCHITECTURES "${GAFIME_CUDA_ARCHITECTURES}"
)

target_compile_options(gafime_cuda_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++23>
)
target_include_directories(gafime_cuda_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
target_link_libraries(gafime_cuda_v1 PRIVATE CUDA::cudart)
