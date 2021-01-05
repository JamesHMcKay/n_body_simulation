#include <iostream>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <assert.h>
#include <vector>
#include <thread>

#include "../include/simulation.cuh"
#include "../include/kernel.cuh"
#include "../include/display.hpp"

using namespace std::chrono;

void Simulation::handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
};

void Simulation::call_kernel_managed(float4 *particles, float3* acceleration, float3* velocities, int N) {
  int blockSize = 1; // UPDATE THIS TO 256 FOR MORE PARTICLES
  int numBlocks = (N + blockSize - 1) / blockSize;
  compute_acceleration<<<numBlocks, blockSize>>>(N, particles, acceleration, physics);
  cudaDeviceSynchronize();
  update_position<<<numBlocks, blockSize>>>(N, particles, acceleration, velocities);
  handle_cuda_error(cudaGetLastError());
  cudaDeviceSynchronize();
}

void Simulation::kernel() {
  IShaderFactory* shaderFactory = new ShaderFactory();

  Display display(SCR_WIDTH, SCR_HEIGHT, shaderFactory);
  display.create_frame_buffers();
  display.load_textures();


  int N = 3;
  printf("%d particles\n", N);


  float3 *velocities;
  float3 *acceleration;
  handle_cuda_error(cudaMallocManaged(&velocities, N * sizeof(float4)));
  handle_cuda_error(cudaMallocManaged(&acceleration, N * sizeof(float4)));

  for (int i = 0; i < N; i++) {
    velocities[i] = make_float3(0.1, 0, 0);
    acceleration[i] = make_float3(0, 0, 0);
  }
  std::cout << "initial values set" << std::endl;

  glm::vec3 cubePositions[] = {
    glm::vec3( 0.0f,  0.0f,  0.0f),
    glm::vec3( 3.0f,  4.0f, -15.0f),
    glm::vec3( 2.0f,  -5.0f, -15.0f)
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
  delete shaderFactory;
}