add_library(gafime_rocm_v1 SHARED
    host.cpp
    tune.cpp
)

target_compile_features(gafime_rocm_v1 PRIVATE cxx_std_23)
target_include_directories(gafime_rocm_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
