add_library(gafime_metal_v1 SHARED
    launcher.mm
    shader.metal
)

target_compile_features(gafime_metal_v1 PRIVATE cxx_std_23)
target_include_directories(gafime_metal_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
set_source_files_properties(shader.metal PROPERTIES HEADER_FILE_ONLY TRUE)
