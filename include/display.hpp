#ifndef DISPLAY_H
#define DISPLAY_H
#include <iostream>
#include <vector>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GL/freeglut.h"
#include "GL/gl.h"

// glm is a header-only library
#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.hpp"
#include "consts.hpp"
#include "shape.hpp"

class Display {
public:
  Display(IShaderFactory *shaderFactory, IShape *shape) : shape(shape) {
    width = SCR_WIDTH;
    height = SCR_HEIGHT;
    init();
    create_window();
    glew_init();
    glEnable(GL_DEPTH_TEST);
    shader = shaderFactory->get_shader("shaders/shader.vs", "shaders/shader.fs");
    shape->create_frame_buffers();
    load_textures();
  }

  void load_textures();

  void main_loop(std::vector<glm::vec3> cubePositions, int num_particles);

  void shutdown();

  bool window_should_close() {
    return glfwWindowShouldClose(window);
  }

private:

  GLFWwindow* window;
  int width;
  int height;

  unsigned int VAO;
  unsigned int VBO;

  unsigned int texture1;

  IShader *shader = nullptr;

  IShape *shape = nullptr;

  void init() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  }

  void create_window();

  void glew_init();
};



#endif