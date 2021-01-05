#ifndef simulation_cuh
#define simulation_cuh

#include <iostream>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <assert.h>
#include <vector>
#include <thread>
#include "glm/glm.hpp"

#include "../include/kernel.cuh"
#include "../include/physics.hpp"
#include "../include/consts.hpp"

__device__ __host__
class Simulation {
public:
  Simulation() {}

  Physics *physics = new Physics();

  void kernel();

  void handle_cuda_error(cudaError_t err);

  void call_kernel_managed(float4 *particles, float3* acceleration, float3* velocities, int N);

  void create_particles();

  void print_details(int id);

private:
  int N = 3;
  float3 *velocities;
  float3 *acceleration;
  float4 *particles;

  std::vector<glm::vec3> get_glm_vec3_particles();


};

#endif