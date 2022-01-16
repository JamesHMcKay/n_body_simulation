#ifndef simulation_cuh
#define simulation_cuh

#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <assert.h>
#include <vector>
#include <thread>
#include <algorithm>
#include "glm/glm.hpp"

#include "../include/kernel.cuh"
#include "../include/physics.hpp"
#include "../include/consts.hpp"
#include "../include/shape.hpp"
#include "../include/display.hpp"

__device__ __host__
class Simulation {
public:
  Simulation() {}

  Physics *physics = new Physics();

  void kernel();

  void handle_cuda_error(cudaError_t err);

  void call_kernel_managed();

  void create_particles();

  void print_details(int id);

  void save_positions();

  void write_output();

private:
  int N = 2;
  float3 *velocities;
  float3 *acceleration;
  float4 *particles;
  int time = 0;

  std::vector<std::vector<float4>> history;

  std::vector<glm::vec3> get_glm_vec3_particles();
};

#endif