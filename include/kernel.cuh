#ifndef kernel_cuh
#define kernel_cuh

#include "../include/kernel.cuh"
#include "../include/physics.hpp"
#include "../include/consts.hpp"

__global__
void compute_acceleration(int N, float4 *particles, float3* acceleration, Physics* physics);

__global__
void update_position(int N, float4 *particles, float3 *acceleration, float3 *velocity);

#endif