cmake_minimum_required(VERSION 3.18)
project(gKDTree VERSION 1.0 LANGUAGES C CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# CUDA Standard  
set(CMAKE_CUDA_STANDARD 14)  # Match C++ standard
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "all-major")

# OpenMP (modern approach)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
endif()

# Include extra libraries
add_subdirectory(3rd_party/cudaKDTree)
add_subdirectory(3rd_party/pybind11)

# First, configure the library with OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(cudaKDTree PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(cudaKDTree PRIVATE OPENMP_FOUND)
endif()

# Python module
pybind11_add_module(_cudaKDTree src/interface.cu)
target_link_libraries(
    _cudaKDTree 
    PRIVATE 
    cudaKDTree
)

