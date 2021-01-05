#ifndef simulation_cuh
#define simulation_cuh

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

};

#endif