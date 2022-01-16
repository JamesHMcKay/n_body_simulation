

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

  // for (int i = 0; i < N; i++) {
  //   velocities[i] = make_float3(0, 0, 0);
  //   acceleration[i] = make_float3(0, 0, 0);
  // }

  float masses_init[] = {100000, 1};

  glm::vec3 velocities_init[] = {
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.2f, 0.0f)
  };

  glm::vec3 acceleration_init[] = {
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f)
  };

  glm::vec3 positions_init[] = {
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(1.5f, 0.0f, 0.0f)
  };

  handle_cuda_error(cudaMallocManaged(&particles, N * sizeof(float4)));
  for (int i = 0; i < N; i++) {
    particles[i] = make_float4(positions_init[i].x, positions_init[i].y, positions_init[i].z, masses_init[i]);
    velocities[i] = make_float3(velocities_init[i].x, velocities_init[i].y, velocities_init[i].z);
    acceleration[i] = make_float3(acceleration_init[i].x, acceleration_init[i].y, acceleration_init[i].z);
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
  while(!display.window_should_close() && time < 10000) {
    print_details(0);
    print_details(1);
    std::cout << " ------ " << std::endl;
    save_positions();
    call_kernel_managed();

    cudaDeviceSynchronize();

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
  std::vector<float4> particles_copy;
  for (int i = 0; i < N; i++) {
    particles_copy.push_back(particles[i]);
  }
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