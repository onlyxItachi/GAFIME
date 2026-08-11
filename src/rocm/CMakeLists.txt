cmake_minimum_required(VERSION 3.26)
project(gafime_rocm_v1 LANGUAGES CXX HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1150")
endif()

set(GAFIME_HIP_WAVE_MI_MODE "64" CACHE STRING
    "MI bin specializations using wave reductions (off, 64, 96, or 64-96)")
set_property(CACHE GAFIME_HIP_WAVE_MI_MODE PROPERTY STRINGS off 64 96 64-96)
if(NOT GAFIME_HIP_WAVE_MI_MODE MATCHES "^(off|64|96|64-96)$")
    message(FATAL_ERROR "GAFIME_HIP_WAVE_MI_MODE must be off, 64, 96, or 64-96")
endif()
set(GAFIME_HIP_WAVE_MI_MASK 0)
if(GAFIME_HIP_WAVE_MI_MODE STREQUAL "64")
    set(GAFIME_HIP_WAVE_MI_MASK 1)
elseif(GAFIME_HIP_WAVE_MI_MODE STREQUAL "96")
    set(GAFIME_HIP_WAVE_MI_MASK 2)
elseif(GAFIME_HIP_WAVE_MI_MODE STREQUAL "64-96")
    set(GAFIME_HIP_WAVE_MI_MASK 3)
endif()

# Precision is a runtime ABI-1.1 selection, not a build-time MI mode.  One
# payload contains all concrete fp32, mixed, and fp64 specialisations.  The
# legacy ABI-1.0 MI marker remains 0 because its float-only entry points remain
# byte-for-byte compatible; ABI-1.1 capability reporting carries the real
# three-profile contract.
set(GAFIME_HIP_PRECISION_PROFILE_MASK 7)
set(GAFIME_HIP_MI_ACCUMULATION_FP64 0)

add_library(gafime_rocm_v1 SHARED
    kernels.hip
    launcher.hip
)
set_target_properties(gafime_rocm_v1 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    HIP_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# HIP emits device-kernel registration metadata and weak C++ template
# instantiations into the host dynamic symbol table even when the source
# visibility defaults are hidden.  The public surface is the explicit C ABI;
# keep only those names in .dynsym so CMake and staged wheel builds enforce the
# same export contract.
set(GAFIME_ROCM_V1_EXPORTS_MAP
    "${CMAKE_CURRENT_LIST_DIR}/../common/gafime_gpu_exports.map")
target_link_options(gafime_rocm_v1 PRIVATE
    "-Wl,--version-script=${GAFIME_ROCM_V1_EXPORTS_MAP}")

target_compile_definitions(gafime_rocm_v1 PRIVATE
    GAFIME_GPU_BUILDING_DLL
    GAFIME_GPU_MI_ACCUMULATION_FP64=${GAFIME_HIP_MI_ACCUMULATION_FP64}
    GAFIME_HIP_PRECISION_PROFILE_MASK=${GAFIME_HIP_PRECISION_PROFILE_MASK}
)

# -O3 is a performance/optimization flag that does not relax IEEE semantics, so it
# is permitted by the "Compiler Ownership" contract without changing numerical
# results (amdclang++/hipcc keep default within-statement FMA contraction).
target_compile_options(gafime_rocm_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--std=c++23>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)
string(REPLACE ";" "," GAFIME_HIP_TARGET_ARCH_TAG "${CMAKE_HIP_ARCHITECTURES}")
set_source_files_properties(kernels.hip PROPERTIES COMPILE_DEFINITIONS
    "GAFIME_HIP_WAVE_MI_MASK=${GAFIME_HIP_WAVE_MI_MASK};GAFIME_HIP_TARGET_ARCH_TAG=\"${GAFIME_HIP_TARGET_ARCH_TAG}\""
)
target_include_directories(gafime_rocm_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

# Link the HIP runtime directly (the packaged hip-config.cmake hardcodes a
# /opt/rocm/bin/hipcc path that is not present on this install).
find_library(GAFIME_AMDHIP64 amdhip64 HINTS $ENV{ROCM_PATH} /opt/rocm PATH_SUFFIXES lib lib64)
if(GAFIME_AMDHIP64)
    target_link_libraries(gafime_rocm_v1 PRIVATE ${GAFIME_AMDHIP64})
endif()

option(GAFIME_ROCM_BUILD_TESTS "Build ROCm v1 ABI smoke tests" OFF)
if(GAFIME_ROCM_BUILD_TESTS)
    include(CTest)
    add_executable(
        gafime_rocm_v1_abi_smoke
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/rocm_v1_abi_smoke.cpp"
    )
    target_compile_features(gafime_rocm_v1_abi_smoke PRIVATE cxx_std_20)
    target_compile_definitions(gafime_rocm_v1_abi_smoke PRIVATE
        GAFIME_EXPECT_MI_ACCUMULATION_FP64=${GAFIME_HIP_MI_ACCUMULATION_FP64}
    )
    target_link_libraries(gafime_rocm_v1_abi_smoke PRIVATE gafime_rocm_v1)
    add_test(NAME gafime_rocm_v1_abi_smoke COMMAND gafime_rocm_v1_abi_smoke)

    add_executable(
        gafime_rocm_precision_abi_smoke
        "${CMAKE_CURRENT_LIST_DIR}/precision_abi_smoke.cpp"
    )
    target_compile_features(gafime_rocm_precision_abi_smoke PRIVATE cxx_std_20)
    target_include_directories(gafime_rocm_precision_abi_smoke PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../common
    )
    target_link_libraries(gafime_rocm_precision_abi_smoke PRIVATE gafime_rocm_v1)
    add_test(NAME gafime_rocm_precision_abi_smoke COMMAND gafime_rocm_precision_abi_smoke)

endif()

option(GAFIME_ROCM_BUILD_BENCHMARKS
    "Build benchmark-only ROCm native timing helpers (never added to CTest)" OFF)
if(GAFIME_ROCM_BUILD_BENCHMARKS)

    # historical pre-freeze typed precision ABI 1.1 baseline
    # is loaded as an independent comparison surface.

    if(WIN32)
        message(FATAL_ERROR
            "ROCm native benchmark helpers are Linux-only evidence producers")
    endif()

    # Native provenance must never depend on PATH or inherited Git redirection
    # variables. Resolve one absolute executable at configure time and bake
    # its path, version, and SHA-256 into every benchmark helper.
    set(GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE "" CACHE FILEPATH
        "Absolute trusted Git executable used for ROCm benchmark provenance")
    if(NOT GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE)
        foreach(candidate "/usr/bin/git" "/bin/git")
            if(IS_EXECUTABLE "${candidate}")
                set(GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE "${candidate}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT IS_ABSOLUTE "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}" OR
       NOT IS_EXECUTABLE "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}")
        message(FATAL_ERROR
            "ROCm benchmark provenance requires an absolute executable Git path")
    endif()
    get_filename_component(GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE
        "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}" REALPATH)
    set(GAFIME_ROCM_BENCHMARK_GIT_ALLOWED FALSE)
    foreach(candidate "/usr/bin/git" "/bin/git")
        if(IS_EXECUTABLE "${candidate}")
            get_filename_component(candidate_realpath "${candidate}" REALPATH)
            if(GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE STREQUAL candidate_realpath)
                set(GAFIME_ROCM_BENCHMARK_GIT_ALLOWED TRUE)
            endif()
        endif()
    endforeach()
    if(NOT GAFIME_ROCM_BENCHMARK_GIT_ALLOWED)
        message(FATAL_ERROR
            "ROCm benchmark Git must resolve to /usr/bin/git or /bin/git")
    endif()
    set(GAFIME_ROCM_BENCHMARK_GIT_CONFIG_NULL "/dev/null")
    execute_process(
        COMMAND "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}" --version
        OUTPUT_VARIABLE GAFIME_ROCM_BENCHMARK_GIT_VERSION
        ERROR_VARIABLE GAFIME_ROCM_BENCHMARK_GIT_VERSION_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GAFIME_ROCM_BENCHMARK_GIT_VERSION_STATUS)
    file(SHA256 "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}"
        GAFIME_ROCM_BENCHMARK_GIT_SHA256)
    if(NOT GAFIME_ROCM_BENCHMARK_GIT_VERSION_STATUS EQUAL 0 OR
       NOT GAFIME_ROCM_BENCHMARK_GIT_VERSION MATCHES "^git version ")
        message(FATAL_ERROR "trusted ROCm benchmark Git executable is unusable")
    endif()

    # Native evidence is built from clean baseline/candidate product roots.
    # The direct helper embeds only the product ROCm kernel translation unit;
    # canonical and host helpers contain no HIP product kernels or device
    # registration metadata, so their fresh processes cannot inherit a direct
    # fatbin merely because a runtime lane flag was supplied.
    set(GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
        "Clean GAFIME product checkout whose ROCm kernels are linked into direct native evidence")
    get_filename_component(GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT
        "${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}" REALPATH)
    set(GAFIME_ROCM_BENCHMARK_PRODUCT_SOURCE
        "${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}/src/rocm/kernels.hip")
    set(GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER
        "${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}/src/rocm/kernels.hpp")
    foreach(required_product_file
            "${GAFIME_ROCM_BENCHMARK_PRODUCT_SOURCE}"
            "${GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER}")
        if(NOT EXISTS "${required_product_file}")
            message(FATAL_ERROR
                "ROCm benchmark product source is missing: ${required_product_file}")
        endif()
    endforeach()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
            --unset=GIT_OBJECT_DIRECTORY --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES
            --unset=GIT_COMMON_DIR --unset=GIT_CEILING_DIRECTORIES
            --unset=GIT_CONFIG --unset=GIT_CONFIG_COUNT
            --unset=GIT_CONFIG_NOSYSTEM --unset=GIT_CONFIG_GLOBAL
            --unset=GIT_CONFIG_SYSTEM --unset=GIT_CONFIG_KEY_0
            --unset=GIT_CONFIG_VALUE_0
            GIT_CONFIG_NOSYSTEM=1
            GIT_CONFIG_GLOBAL=${GAFIME_ROCM_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_SYSTEM=${GAFIME_ROCM_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_COUNT=0
            "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}" -C
            "${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}" rev-parse HEAD
        OUTPUT_VARIABLE GAFIME_ROCM_BENCHMARK_PRODUCT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GAFIME_ROCM_BENCHMARK_PRODUCT_GIT_STATUS
    )
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
            --unset=GIT_OBJECT_DIRECTORY --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES
            --unset=GIT_COMMON_DIR --unset=GIT_CEILING_DIRECTORIES
            --unset=GIT_CONFIG --unset=GIT_CONFIG_COUNT
            --unset=GIT_CONFIG_NOSYSTEM --unset=GIT_CONFIG_GLOBAL
            --unset=GIT_CONFIG_SYSTEM --unset=GIT_CONFIG_KEY_0
            --unset=GIT_CONFIG_VALUE_0
            GIT_CONFIG_NOSYSTEM=1
            GIT_CONFIG_GLOBAL=${GAFIME_ROCM_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_SYSTEM=${GAFIME_ROCM_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_COUNT=0
            "${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}" -C
            "${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}"
            status --porcelain=v1 --untracked-files=all
        OUTPUT_VARIABLE GAFIME_ROCM_BENCHMARK_PRODUCT_TREE_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GAFIME_ROCM_BENCHMARK_PRODUCT_TREE_GIT_STATUS
    )
    string(LENGTH "${GAFIME_ROCM_BENCHMARK_PRODUCT_COMMIT}"
        GAFIME_ROCM_BENCHMARK_PRODUCT_COMMIT_LENGTH)
    if(NOT GAFIME_ROCM_BENCHMARK_PRODUCT_GIT_STATUS EQUAL 0 OR
       NOT GAFIME_ROCM_BENCHMARK_PRODUCT_COMMIT_LENGTH EQUAL 40 OR
       NOT GAFIME_ROCM_BENCHMARK_PRODUCT_COMMIT MATCHES "^[0-9a-fA-F]+$")
        message(FATAL_ERROR
            "ROCm benchmark product root must resolve to a full Git commit")
    endif()
    if(NOT GAFIME_ROCM_BENCHMARK_PRODUCT_TREE_GIT_STATUS EQUAL 0 OR
       NOT GAFIME_ROCM_BENCHMARK_PRODUCT_TREE_STATUS STREQUAL "")
        message(FATAL_ERROR
            "ROCm benchmark product root must be a clean Git worktree")
    endif()
    file(SHA256 "${GAFIME_ROCM_BENCHMARK_PRODUCT_SOURCE}"
        GAFIME_ROCM_BENCHMARK_PRODUCT_SOURCE_SHA256)
    file(SHA256 "${GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER}"
        GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER_SHA256)
    # The helper source is owned by this (common) harness checkout, not by the
    # clean product root.  A baseline product root intentionally does not
    # contain the candidate's lane harness.
    set(GAFIME_ROCM_BENCHMARK_DIRECT_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/rocm_native_direct_lane.hip")
    file(SHA256 "${GAFIME_ROCM_BENCHMARK_DIRECT_SOURCE}"
        GAFIME_ROCM_BENCHMARK_DIRECT_SOURCE_SHA256)

    # The clean 8df product predates the current selected-row kernel spelling
    # and the extra MI legacy-nonfinite argument.  Select the direct wrapper's
    # declaration-compatible call surface from the product header while still
    # compiling the exact product source translation unit.
    file(READ "${GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER}"
        GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER_TEXT)
    string(FIND "${GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER_TEXT}"
        "copy_selected_rows_kernel" GAFIME_ROCM_BENCHMARK_NEW_KERNEL_API_OFFSET)
    if(GAFIME_ROCM_BENCHMARK_NEW_KERNEL_API_OFFSET GREATER -1)
        set(GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API 2)
    else()
        set(GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API 1)
    endif()

    set(GAFIME_ROCM_NATIVE_TIMING_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/rocm_native_timing.cpp")
    set(GAFIME_ROCM_NATIVE_TIMING_DIRECT_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/rocm_native_direct_lane.hip")

    function(gafime_rocm_add_native_timing_target
        target_name lane_value lane_name include_product_kernels)
        set(native_sources "${GAFIME_ROCM_NATIVE_TIMING_SOURCE}")
        if(include_product_kernels)
            list(APPEND native_sources
                "${GAFIME_ROCM_BENCHMARK_PRODUCT_SOURCE}"
                "${GAFIME_ROCM_NATIVE_TIMING_DIRECT_SOURCE}")
        endif()
        add_executable(${target_name} ${native_sources})
        target_compile_features(${target_name} PRIVATE cxx_std_20)
        set_target_properties(${target_name} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
        target_compile_definitions(${target_name} PRIVATE
            __HIP_PLATFORM_AMD__
            GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE=${lane_value}
            GAFIME_ROCM_NATIVE_TIMING_COMPILED_LANE_NAME="${lane_name}"
            GAFIME_GPU_MI_ACCUMULATION_FP64=${GAFIME_HIP_MI_ACCUMULATION_FP64}
            GAFIME_HIP_PRECISION_PROFILE_MASK=${GAFIME_HIP_PRECISION_PROFILE_MASK}
            GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE="${GAFIME_ROCM_BENCHMARK_GIT_EXECUTABLE}"
            GAFIME_NATIVE_TRUSTED_GIT_SHA256="${GAFIME_ROCM_BENCHMARK_GIT_SHA256}"
            GAFIME_NATIVE_TRUSTED_GIT_VERSION="${GAFIME_ROCM_BENCHMARK_GIT_VERSION}"
        )
        if(include_product_kernels)
            target_compile_definitions(${target_name} PRIVATE
                GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_ROOT="${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}"
                GAFIME_ROCM_NATIVE_TIMING_LINKED_PRODUCT_COMMIT="${GAFIME_ROCM_BENCHMARK_PRODUCT_COMMIT}"
                GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_SHA256="${GAFIME_ROCM_BENCHMARK_PRODUCT_SOURCE_SHA256}"
                GAFIME_ROCM_NATIVE_TIMING_LINKED_KERNELS_HEADER_SHA256="${GAFIME_ROCM_BENCHMARK_PRODUCT_HEADER_SHA256}"
                GAFIME_ROCM_NATIVE_TIMING_LINKED_DIRECT_SOURCE_SHA256="${GAFIME_ROCM_BENCHMARK_DIRECT_SOURCE_SHA256}"
                GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API=${GAFIME_ROCM_NATIVE_PRODUCT_KERNEL_API})
        endif()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:--std=c++23>
            $<$<COMPILE_LANGUAGE:HIP>:-O3>
        )
        target_include_directories(${target_name} PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/../common"
            "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu"
        )
        if(include_product_kernels)
            target_include_directories(${target_name} PRIVATE
                "${GAFIME_ROCM_BENCHMARK_PRODUCT_ROOT}/src/rocm")
        endif()
        # Keep the HIP runtime linked for HIP-event timing and dlopen support,
        # but never link the product shared object into any timing helper.
        target_link_libraries(${target_name} PRIVATE
            ${GAFIME_AMDHIP64}
            ${CMAKE_DL_LIBS}
        )
    endfunction()

    # Preserve the historical direct target name for runner configurations.
    gafime_rocm_add_native_timing_target(
        gafime_rocm_native_timing 1 supplemental_internal_kernel TRUE)
    gafime_rocm_add_native_timing_target(
        gafime_rocm_native_timing_canonical 2 canonical_payload_api FALSE)
    gafime_rocm_add_native_timing_target(
        gafime_rocm_native_timing_host 3 supplemental_host_phase FALSE)

endif()

if(GAFIME_ROCM_BUILD_TESTS)
    set(GAFIME_ABI_CONSUMER_PAYLOAD_TARGET gafime_rocm_v1)
    set(GAFIME_ABI_CONSUMER_BACKEND_KIND 3)
    set(GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT 3)
    set(GAFIME_ABI_CONSUMER_TEST_SUFFIX rocm)
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
