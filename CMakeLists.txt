cmake_minimum_required(VERSION 3.1)

project(onnx_dynamic_load CXX)

# Build with -std=c++11
set(CMAKE_CXX_STANDARD  11)

set(THIRD_PARTY_PATH "${PROJECT_SOURCE_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")


# Build protobuf
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/protobuf/cmake)

# Build ONNX
set(BUILD_ONNX_PYTHON OFF CACHE BOOL "Build Python binaries")
set(ONNX_GEN_PB_TYPE_STUBS OFF CACHE BOOL "Generate protobuf python type stubs")
set(ONNX_ML ON CACHE BOOL "Enable traditional ML API")
add_subdirectory(${THIRD_PARTY_PATH}/onnx)

target_compile_options(onnx PRIVATE "-Wno-unused-parameter")
target_compile_options(onnx_proto PRIVATE "-Wno-unused-parameter")
target_compile_options(onnx PRIVATE "-Wno-unused-but-set-variable")
target_compile_options(onnx_proto PRIVATE "-Wno-unused-but-set-variable")

# Build onnx dynamic loader
include_directories(include)
include_directories(${ONNX_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS "${PROJECT_SOURCE_DIR}/include/*.h"
                       "${PROJECT_SOURCE_DIR}/src/*.cc")

add_executable(onnx_dynamic_loader.out ${SRCS})
add_dependencies(onnx_dynamic_loader.out onnx onnx_proto)
target_link_libraries(onnx_dynamic_loader.out PRIVATE onnx dl)
