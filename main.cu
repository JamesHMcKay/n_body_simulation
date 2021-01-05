#include <iostream>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GL/freeglut.h"
#include "GL/gl.h"
#include "glm/glm.hpp"

#include "include/simulation.cuh"
#include "include/kernel.cuh"


int main(int argc, char **argv) {
  Simulation particle_simulation;
  particle_simulation.kernel();

  return 0;
}
