cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(n_body_simulation LANGUAGES CXX CUDA)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

set(
  SOURCE_FILES
  src/shader.cpp
  src/display.cpp
  src/kernel.cu
  src/simulation.cu
  include/kernel.cuh
  include/display.hpp
  include/shader.hpp
  include/stb_image.h
  include/physics.hpp
  include/consts.hpp
  include/simulation.cuh
)

add_executable(
  simulation
  main.cu
  ${SOURCE_FILES}
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

add_executable(run_tests
  tests/test_compute.cu
  tests/test_display.cpp
  tests/mocks/
  ${SOURCE_FILES}
)

target_include_directories (
    run_tests
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/mocks/
)

target_compile_features(run_tests PUBLIC cxx_std_14)

target_link_libraries(run_tests glfw)
target_link_libraries(run_tests OpenGL::GL)
target_link_libraries(run_tests GLEW)
target_link_libraries(run_tests gtest_main gmock_main)

set_target_properties(run_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
add_test(NAME run_tests COMMAND run_tests)