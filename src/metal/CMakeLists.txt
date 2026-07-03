cmake_minimum_required(VERSION 3.26)
project(gafime_metal_v1 LANGUAGES CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "gafime_metal_v1 requires Apple Metal and Objective-C++.")
endif()

find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

set(GAFIME_METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/gafime_metal_v1.air")
set(GAFIME_METAL_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/gafime_metal_v1.metallib")

add_custom_command(
    OUTPUT "${GAFIME_METAL_LIBRARY}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -c "${CMAKE_CURRENT_LIST_DIR}/shader.metal"
        -o "${GAFIME_METAL_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
        "${GAFIME_METAL_AIR}"
        -o "${GAFIME_METAL_LIBRARY}"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/shader.metal"
    VERBATIM
)

add_custom_target(gafime_metal_v1_metallib DEPENDS "${GAFIME_METAL_LIBRARY}")

add_library(gafime_metal_v1 SHARED
    launcher.mm
    shader.metal
)

add_dependencies(gafime_metal_v1 gafime_metal_v1_metallib)

target_compile_features(gafime_metal_v1 PRIVATE cxx_std_23)
target_compile_definitions(gafime_metal_v1 PRIVATE
    GAFIME_METAL_DEFAULT_LIBRARY_PATH="${GAFIME_METAL_LIBRARY}"
)
target_include_directories(gafime_metal_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
target_link_libraries(gafime_metal_v1 PRIVATE
    "-framework Foundation"
    "-framework Metal"
)
set_source_files_properties(shader.metal PROPERTIES HEADER_FILE_ONLY TRUE)
