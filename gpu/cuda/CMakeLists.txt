cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(gafime_cuda_v1 SHARED
    device.cu
    host.cpp
    tune.cpp
)

set_source_files_properties(host.cpp PROPERTIES LANGUAGE CUDA)

# Codegen policy (ROADMAP §2). Default to the live RTX 4060 (sm_89) but allow an
# override, and always emit PTX for forward-compat. No --use_fast_math: FTZ +
# approximate transcendentals would break the f64-accumulator golden parity.
if(NOT DEFINED GAFIME_CUDA_ARCHITECTURES)
    set(GAFIME_CUDA_ARCHITECTURES "89-real;89-virtual")
endif()
set_target_properties(gafime_cuda_v1 PROPERTIES
    CUDA_ARCHITECTURES "${GAFIME_CUDA_ARCHITECTURES}"
)

target_compile_options(gafime_cuda_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++23>
    # -O3 host side, -Xptxas -O3 device side, + line info for profiling (not -G).
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
)
target_include_directories(gafime_cuda_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
target_link_libraries(gafime_cuda_v1 PRIVATE CUDA::cudart)
