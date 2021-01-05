

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

void Simulation::create_particles() {
  printf("%d particles\n", N);

  handle_cuda_error(cudaMallocManaged(&velocities, N * sizeof(float4)));
  handle_cuda_error(cudaMallocManaged(&acceleration, N * sizeof(float4)));

  for (int i = 0; i < N; i++) {
    velocities[i] = make_float3(0.1, 0, 0);
    acceleration[i] = make_float3(0, 0, 0);
  }
  std::cout << "initial values set" << std::endl;

  float masses[] = {10, 10, 4};

  glm::vec3 cubePositions[] = {
    glm::vec3( 0.0f,  0.0f,  0.0f),
    glm::vec3( 3.0f,  4.0f, -15.0f),
    glm::vec3( 2.0f,  -5.0f, -15.0f)
  };

  handle_cuda_error(cudaMallocManaged(&particles, N * sizeof(float4)));
  for (int i = 0; i < N; i++) {
    particles[i] = make_float4(cubePositions[i].x, cubePositions[i].y, cubePositions[i].z, masses[i]);
  }
}

void Simulation::print_details(int id) {
  printf("Particle %d location = (%f, %f, %f)\n", id, particles[0].x, particles[0].y, particles[0].z);
  printf("Particle %d acceleration = (%f, %f, %f)\n", id, acceleration[0].x, acceleration[0].y, acceleration[0].z);
  printf("Particle %d velocity = (%f, %f, %f)\n", id, velocities[0].x, velocities[0].y, velocities[0].z);
}

void Simulation::kernel() {
  IShaderFactory* shaderFactory = new ShaderFactory();

  Display display(SCR_WIDTH, SCR_HEIGHT, shaderFactory);
  display.create_frame_buffers();
  display.load_textures();

  create_particles();
  print_details(1);
  while(!display.window_should_close()) {
    call_kernel_managed(particles, acceleration, velocities, N);
    std::cout << " ------ " << std::endl;
    cudaDeviceSynchronize();
    print_details(1);
    using namespace std::chrono_literals;
    auto start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(100ms);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end-start;
    std::cout << "Waited " << elapsed.count() << " ms\n";
    display.main_loop(get_glm_vec3_particles(), N) ;
  }
  cudaFree(particles);

  display.shutdown();
  delete shaderFactory;
}

std::vector<glm::vec3> Simulation::get_glm_vec3_particles() {
  std::vector<glm::vec3> result;
  for (int i = 0; i < N; i++) {
    result.push_back(glm::vec3(particles[i].x, particles[i].y, particles[i].z));
  }
  return result;
}