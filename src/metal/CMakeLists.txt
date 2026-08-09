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
        -fno-fast-math
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
set_target_properties(gafime_metal_v1 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OBJCXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

add_dependencies(gafime_metal_v1 gafime_metal_v1_metallib)

target_compile_features(gafime_metal_v1 PRIVATE cxx_std_23)
# -O3 is a performance/optimization flag that does not relax IEEE semantics, so it
# is permitted by the "Compiler Ownership" contract without changing numerical
# results. Applies to the launcher.mm host path; shader.metal is compiled by the
# Metal toolchain (HEADER_FILE_ONLY here). The offline shader compilation above
# explicitly disables fast math so fp32 histogram boundaries, non-finite
# handling, and visible scores retain the declared arithmetic contract.
target_compile_options(gafime_metal_v1 PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)
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

option(GAFIME_METAL_BUILD_TESTS "Build Metal ABI consumer tests" OFF)
if(GAFIME_METAL_BUILD_TESTS)
    include(CTest)
    set(GAFIME_ABI_CONSUMER_PAYLOAD_TARGET gafime_metal_v1)
    set(GAFIME_ABI_CONSUMER_BACKEND_KIND 4)
    set(GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT 1)
    set(GAFIME_ABI_CONSUMER_TEST_SUFFIX metal)
    set(GAFIME_ABI_CONSUMER_ENABLE_DYNAMIC_ARITY ON)
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/abi_consumers"
        "${CMAKE_CURRENT_BINARY_DIR}/abi_consumers"
    )
    unset(GAFIME_ABI_CONSUMER_PAYLOAD_TARGET)
    unset(GAFIME_ABI_CONSUMER_BACKEND_KIND)
    unset(GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT)
    unset(GAFIME_ABI_CONSUMER_TEST_SUFFIX)
    unset(GAFIME_ABI_CONSUMER_ENABLE_DYNAMIC_ARITY)
endif()

option(
    GAFIME_METAL_BUILD_BENCHMARKS
    "Build test-only Metal native timing/decomposition evidence tools"
    OFF
)
if(GAFIME_METAL_BUILD_BENCHMARKS)
    # This executable is an evidence fixture, not part of the payload and not
    # an ordinary correctness-oriented CTest. Its direct metallib lane is
    # supplemental GPUStartTime/GPUEndTime evidence; the helper also loads the
    # exact wheel payload at runtime and exercises canonical ABI 1.1 without
    # adding instrumentation to product code.
    add_executable(
        gafime_metal_precision_native_timing
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/metal_precision_native_timing.mm"
    )
    add_dependencies(
        gafime_metal_precision_native_timing
        gafime_metal_v1_metallib
    )
    target_compile_features(
        gafime_metal_precision_native_timing
        PRIVATE cxx_std_20
    )
    target_compile_options(
        gafime_metal_precision_native_timing
        PRIVATE -O3 -fobjc-arc
    )
    target_compile_definitions(
        gafime_metal_precision_native_timing
        PRIVATE
            GAFIME_METAL_TIMING_DEFAULT_LIBRARY_PATH="${GAFIME_METAL_LIBRARY}"
            GAFIME_METAL_TIMING_SHADER_SOURCE="${CMAKE_CURRENT_LIST_DIR}/shader.metal"
    )
    target_link_libraries(
        gafime_metal_precision_native_timing
        PRIVATE
            "-framework Foundation"
            "-framework Metal"
    )
endif()
