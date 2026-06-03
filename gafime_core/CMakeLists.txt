cmake_minimum_required(VERSION 3.20)
project(gafime_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GAFIME_CORE_USE_FETCHCONTENT "Fetch pybind11 if not found" ON)
option(GAFIME_CORE_ENABLE_OPENMP "Enable OpenMP parallel loops" OFF)
option(GAFIME_CORE_USE_DOUBLE_PRECISION "Build C++ Core with fp64 real_t instead of default fp32" OFF)

# Configure macOS framework search to prioritize virtual environments
set(CMAKE_FIND_FRAMEWORK LAST)

# Try to find Development.Module first (required for linux manylinux wheels)
# If it fails (e.g. on macOS where bundled pythons might lack the split Module tag), fallback to Development
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if (NOT Python3_FOUND)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND AND GAFIME_CORE_USE_FETCHCONTENT)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

if (NOT pybind11_FOUND AND NOT TARGET pybind11::module)
  message(FATAL_ERROR "pybind11 not found. Install pybind11 or set GAFIME_CORE_USE_FETCHCONTENT=ON.")
endif()

set(GAFIME_CORE_SOURCES
  src/gafime_core.cpp
  src/simd_scalar.cpp
  src/simd_dispatch.cpp
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" GAFIME_SYSTEM_PROCESSOR)
if (GAFIME_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i[3-6]86")
  list(APPEND GAFIME_CORE_SOURCES
    src/simd_x86_sse42.cpp
    src/simd_x86_avx2.cpp
    src/simd_x86_avx512.cpp
  )
  if (MSVC)
    set_source_files_properties(src/simd_x86_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/simd_x86_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/simd_x86_sse42.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2")
    set_source_files_properties(src/simd_x86_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/simd_x86_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
  endif()
endif()

if (GAFIME_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  list(APPEND GAFIME_CORE_SOURCES src/simd_arm_neon.cpp)
endif()

pybind11_add_module(gafime_core ${GAFIME_CORE_SOURCES})
target_include_directories(gafime_core PRIVATE ${Python3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gafime_core PRIVATE cxx_std_23)

if (GAFIME_CORE_USE_DOUBLE_PRECISION)
  target_compile_definitions(gafime_core PRIVATE GAFIME_USE_DOUBLE_PRECISION=1)
endif()

if (MSVC)
  target_compile_options(gafime_core PRIVATE /O2 /EHsc)
else()
  target_compile_options(gafime_core PRIVATE -O3)
endif()

if (GAFIME_CORE_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(gafime_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(gafime_core PRIVATE GAFIME_CORE_OPENMP=1)
  endif()
endif()
