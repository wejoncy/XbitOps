cmake_minimum_required(VERSION 3.21)
#project(gemv LANGUAGES CXX CUDA) 
project(gemv)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)   


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(SRC_LIST main.cc unpack_weight_k16.cu cpp_simulate.cc)
#aux_source_directory(src SRC_LIST)
include_directories(./)

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} cublas curand cuda)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70")
