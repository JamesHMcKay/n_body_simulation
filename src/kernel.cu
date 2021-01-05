#include <iostream>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <assert.h>
#include <vector>
#include <thread>

#include "../include/kernel.cuh"

#include "../include/display.hpp"

using namespace std::chrono;

__global__
void compute_acceleration(int N, float4 *particles, float3* acceleration, Physics* physics) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = id; i < N; i += blockDim.x * gridDim.x) {
    float3 acc_temp = make_float3(0, 0, 0);
    for (int j = 0; j < N; j++) {
      if (j != i) {
        acc_temp = physics->add_acceleration(particles[i], particles[j], acc_temp);
      }
      
    }
    acceleration[i] = acc_temp;
  }
};

__global__
void update_position(int N, float4 *particles, float3 *acceleration, float3 *velocity) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float delta_t = 1;

  for (int i = id; i < N; i += blockDim.x * gridDim.x) {
    float3 s;
    s.x = velocity[i].x + 0.5 * acceleration[i].x * delta_t;
    s.y = velocity[i].y + 0.5 * acceleration[i].y * delta_t;
    s.z = velocity[i].z + 0.5 * acceleration[i].z * delta_t;

    particles[i] = make_float4(
      particles[i].x + s.x * delta_t,
      particles[i].y + s.y * delta_t,
      particles[i].z + s.z * delta_t,
      particles[i].w
    );

    // this isn't quite right, the acceleration here should be i + 1
    float3 new_vel;
    new_vel.x = velocity[i].x + 0.5 * acceleration[i].x * delta_t;
    new_vel.y = velocity[i].y + 0.5 * acceleration[i].y * delta_t;
    new_vel.z = velocity[i].z + 0.5 * acceleration[i].z * delta_t;
    velocity[i] = new_vel;
  }
}
