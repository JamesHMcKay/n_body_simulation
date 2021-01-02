#include <iostream>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <assert.h>
#include <vector>
#include "../include/kernel.cuh"
#include <thread>

#include "../include/display.hpp"

using namespace std::chrono;

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

#define SOFTENING 0.1f
#define BLOCK_SIZE 256
#define G 1.0f

void handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__device__
float3 add_acceleration(float4 p1, float4 p2, float3 a) {
  float3 delta;
  delta.x = (p1.x - p2.x);
  delta.y = (p1.y - p2.y);
  delta.z = (p1.z - p2.z);

  float r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z + SOFTENING;
  float inv_r2 = p2.w / sqrtf(r2 * r2 * r2);

  a.x = - G * delta.x * inv_r2 + a.x;
  a.y = - G * delta.y * inv_r2 + a.y;
  a.z = - G * delta.z * inv_r2 + a.z;
  return a;
}

__global__
void compute_acceleration(int N, float4 *particles, float3* acceleration) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = id; i < N; i += blockDim.x * gridDim.x) {
    float3 acc_temp = make_float3(0, 0, 0);
    for (int j = 0; j < N; j++) {
      if (j != i) {
        acc_temp = add_acceleration(particles[i], particles[j], acc_temp);
      }
      
    }
    acceleration[i] = acc_temp;
  }
}

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

void call_kernel_managed(float4 *particles, float3* acceleration, float3* velocities, int N) {
  int blockSize = 1; // UPDATE THIS TO 256 FOR MORE PARTICLES
  int numBlocks = (N + blockSize - 1) / blockSize;

  compute_acceleration<<<numBlocks, blockSize>>>(N, particles, acceleration);
  cudaDeviceSynchronize();
  update_position<<<numBlocks, blockSize>>>(N, particles, acceleration, velocities);
  handle_cuda_error(cudaGetLastError());
  cudaDeviceSynchronize();
}

void kernel() {
  Display display(SCR_WIDTH, SCR_HEIGHT);
  display.create_frame_buffers();
  display.load_textures();


  int N = 3;
  printf("%d particles\n", N);


  float3 *velocities;
  float3 *acceleration;
  handle_cuda_error(cudaMallocManaged(&velocities, N * sizeof(float4)));
  handle_cuda_error(cudaMallocManaged(&acceleration, N * sizeof(float4)));

  for (int i = 0; i < N; i++) {
    // particles[i] = make_float4(sin(i), cos(i), 1, 1);
    velocities[i] = make_float3(0.1, 0, 0);
    acceleration[i] = make_float3(0, 0, 0);
  }
  std::cout << "initial values set" << std::endl;

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // cudaEventRecord(start);


  glm::vec3 cubePositions[] = {
    glm::vec3( 0.0f,  0.0f,  0.0f),
    glm::vec3( 3.0f,  4.0f, -15.0f),
    glm::vec3( 2.0f,  -5.0f, -15.0f)
    // glm::vec3(-1.5f, -2.2f, -2.5f),
    // glm::vec3(-3.8f, -2.0f, -12.3f),
    // glm::vec3( 9.4f, -0.4f, -3.5f),
    // glm::vec3(-1.7f,  3.0f, -7.5f),
    // glm::vec3( 1.3f, -2.0f, -2.5f),
    // glm::vec3( 1.5f,  2.0f, -2.5f),
    // glm::vec3( 1.5f,  0.2f, -1.5f),
    // glm::vec3(-1.3f,  1.0f, -1.5f)
  };
  float masses[] = {10, 10, 4};

  float4 *particles;
  handle_cuda_error(cudaMallocManaged(&particles, N * sizeof(float4)));
  for (int i = 0; i < N; i++) {
    particles[i] = make_float4(cubePositions[i].x, cubePositions[i].y, cubePositions[i].z, masses[i]);
  }
  std::cout << "particle 1 before: " << particles[0].x << " " << particles[0].y << " " << particles[0].z << std::endl;
  std::cout << "acceleration 1 before: " << acceleration[0].x << " " << acceleration[1].y << " " << acceleration[0].z << std::endl;
  std::cout << "velocities 1 before: " << velocities[0].x << " " << velocities[0].y << " " << velocities[0].z << std::endl;
  while(!display.window_should_close()) {

    call_kernel_managed(particles, acceleration, velocities, N);
    std::cout << " ------ " << std::endl;
    cudaDeviceSynchronize();
    std::cout << "particle 1 after: " << particles[0].x << " " << particles[0].y << " " << particles[0].z << std::endl;
    std::cout << "acceleration 1 after: " << acceleration[0].x << " " << acceleration[0].y << " " << acceleration[0].z << std::endl;
    std::cout << "velocities 1 after: " << velocities[0].x << " " << velocities[0].y << " " << velocities[0].z << std::endl;
    
    using namespace std::chrono_literals;
    auto start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(100ms);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end-start;
    std::cout << "Waited " << elapsed.count() << " ms\n";
    for (int i = 0; i < N; i++) {
      cubePositions[i] = glm::vec3(particles[i].x, particles[i].y, particles[i].z);
    }
    display.main_loop(cubePositions, N) ;
  }
  cudaFree(particles);

  display.shutdown();



}