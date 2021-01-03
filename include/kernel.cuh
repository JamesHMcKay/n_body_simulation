#ifndef kernel_cuh
#define kernel_cuh

#include "../include/kernel.cuh"

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

#define SOFTENING 0.1f
#define BLOCK_SIZE 256
#define G 1.0f


void kernel();

void handle_cuda_error(cudaError_t err);

__device__ __host__
float3 add_acceleration(float4 p1, float4 p2, float3 a);

__global__
void compute_acceleration(int N, float4 *particles, float3* acceleration);

#endif