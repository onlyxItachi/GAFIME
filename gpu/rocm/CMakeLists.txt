cmake_minimum_required(VERSION 3.26)
project(gafime_rocm_v1 LANGUAGES CXX HIP)

add_library(gafime_rocm_v1 SHARED
    host.cpp
    tune.cpp
)

set_source_files_properties(host.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(gafime_rocm_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23>
    $<$<COMPILE_LANGUAGE:HIP>:--std=c++23>
)
target_include_directories(gafime_rocm_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
