cmake_minimum_required(VERSION 3.18)
project(kernel_sdf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# A10 = Ampere = sm_86. Default if user didn't override.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(prep_loader STATIC src/prep_loader.cpp)
target_include_directories(prep_loader PUBLIC src)

add_executable(hello src/hello.cu)

add_executable(dump_prep src/dump_prep.cpp)
target_link_libraries(dump_prep PRIVATE prep_loader)

find_package(CUDAToolkit REQUIRED)

add_executable(omega_linear src/omega_linear.cu)
target_link_libraries(omega_linear PRIVATE prep_loader CUDA::cudart CUDA::cublas)

add_executable(omega_rbf src/omega_rbf.cu)
target_link_libraries(omega_rbf PRIVATE prep_loader CUDA::cudart CUDA::cublas)

add_executable(omega_rbf_opt src/omega_rbf_opt.cu)
target_link_libraries(omega_rbf_opt PRIVATE prep_loader CUDA::cudart CUDA::cublas)
