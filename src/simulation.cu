

#include "../include/simulation.cuh"

using namespace std::chrono;

void Simulation::handle_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
};

void Simulation::call_kernel_managed() {
  int blockSize = 1; // UPDATE THIS TO 256 FOR MORE PARTICLES
  int numBlocks = (N + blockSize - 1) / blockSize;
  compute_acceleration<<<numBlocks, blockSize>>>(N, particles, acceleration, physics);
  cudaDeviceSynchronize();
  update_position<<<numBlocks, blockSize>>>(N, particles, acceleration, velocities);
  time = time + 1;
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
    glm::vec3( 0.1f, 0.1f, 0.1f),
    glm::vec3( 3.0f, 4.0f, -15.0f),
    glm::vec3( 2.0f, -5.0f, -15.0f)
  };

  handle_cuda_error(cudaMallocManaged(&particles, N * sizeof(float4)));
  for (int i = 0; i < N; i++) {
    particles[i] = make_float4(cubePositions[i].x, cubePositions[i].y, cubePositions[i].z, masses[i]);
  }
}

void Simulation::print_details(int id) {
  printf("Particle %d location = (%f, %f, %f)\n", id, particles[id].x, particles[id].y, particles[id].z);
  printf("Particle %d acceleration = (%f, %f, %f)\n", id, acceleration[id].x, acceleration[id].y, acceleration[id].z);
  printf("Particle %d velocity = (%f, %f, %f)\n", id, velocities[id].x, velocities[id].y, velocities[id].z);
}

void Simulation::kernel() {
  IShaderFactory* shaderFactory = new ShaderFactory();
  IShape* box = new Box();

  Display display(shaderFactory, box);

  create_particles();
  while(!display.window_should_close() && time < 200) {
    print_details(1);
    save_positions();
    call_kernel_managed();
    std::cout << " ------ " << std::endl;
    cudaDeviceSynchronize();
    using namespace std::chrono_literals;
    auto start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(100ms);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Waited " << elapsed.count() << " ms\n";
    display.main_loop(get_glm_vec3_particles(), N) ;
  }
  cudaFree(particles);

  display.shutdown();
  write_output();
  delete shaderFactory;
  delete box;
}

std::vector<glm::vec3> Simulation::get_glm_vec3_particles() {
  std::vector<glm::vec3> result;
  for (int i = 0; i < N; i++) {
    result.push_back(glm::vec3(particles[i].x, particles[i].y, particles[i].z));
  }
  return result;
}

void Simulation::save_positions() {
  // output table has columns time, particle, x, y, z
  int id = 1;
  std::vector<float4> particles_copy;
  for (int i = 0; i < N; i++) {
    particles_copy.push_back(particles[i]);
  }
  printf("particles_copy %d location = (%f, %f, %f)\n", id, particles_copy[id].x, particles_copy[id].y, particles_copy[id].z);
  history.push_back(particles_copy);
}

void Simulation::write_output() {
  std::ofstream output_file;
  output_file.open("saved_positions.csv");
  output_file << "time,id,x,y,z\n";
  for (int i = 0; i < history.size(); i++) {
    for (int j = 0; j < N; j++) {
      output_file << i << "," << j << "," << history[i][j].x << "," << history[i][j].y << "," << history[i][j].z << "\n";
    }
  }

  output_file.close();
}