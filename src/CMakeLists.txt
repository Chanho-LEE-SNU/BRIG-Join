cmake_minimum_required(VERSION 3.25)
project(test)



set(CMAKE_CXX_STANDARD 20)
set(GCC_COVERAGE_COMPILE_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread -lpthread -O3 -fopenmp -march=native")


add_executable(test driver.cpp lock.h hash_kernel.cu hash_kernel.h kernel_functions.h kernel_functions.cu)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
endif()

