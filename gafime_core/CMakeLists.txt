cmake_minimum_required(VERSION 3.20)
project(gafime_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GAFIME_CORE_USE_FETCHCONTENT "Fetch pybind11 if not found" ON)
option(GAFIME_CORE_ENABLE_OPENMP "Enable OpenMP parallel loops" OFF)

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

pybind11_add_module(gafime_core src/gafime_core.cpp)
target_compile_features(gafime_core PRIVATE cxx_std_23)

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
