cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cpp_libdmet)

set(anaconda "/home/mitra054/.conda/envs/msi2")

# Set INCLUDE and LIB 
LINK_DIRECTORIES(/home/mitra054/.conda/envs/msi2/lib)
INCLUDE_DIRECTORIES(/home/mitra054/.conda/envs/msi2/include)

# Find the LAWrap for mkl_blas/lapack, pybind11
find_package(LAWrap CONFIG REQUIRED)
message(STATUS "Found LAWrap: ${LAWrap_CONFIG}")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11: ${pybind11_CONFIG}")

# Set python
set(PYTHON_EXECUTABLE)

# Openmp
include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set pybind11 c++ standard
set(PYBIND11_CPP_STANDARD -std=c++11)

# Creates a python module named "module_name"
pybind11_add_module(libdmet MODULE libdmet.cpp)
target_link_libraries(libdmet PUBLIC mkl_intel_lp64 mkl_sequential mkl_core)