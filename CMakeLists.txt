cmake_minimum_required(VERSION 3.23)
project(Learn-to-Prune)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/hnswlib)
include_directories(include/ivf)


find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()


add_definitions(-std=c++14 -O3)
add_subdirectory(src)