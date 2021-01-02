cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(n_body_simulation LANGUAGES CXX CUDA)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_executable(
  simulation
  src/shader.cpp
  src/display.cpp
  src/kernel.cu
  include/kernel.cuh
  include/display.hpp
  include/shader.hpp
  include/stb_image.h
  main.cu
)

target_link_libraries(simulation glfw)
target_link_libraries(simulation OpenGL::GL)
target_link_libraries(simulation GLEW)

target_compile_features(simulation PUBLIC cxx_std_14)

set_target_properties(simulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build tests
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

add_executable(test_simulation
  tests/test_utils.cpp
)

target_link_libraries(test_simulation PRIVATE gtest_main)

enable_testing()
add_test(NAME test_simulation COMMAND test_simulation)