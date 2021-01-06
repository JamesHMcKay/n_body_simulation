#include <iostream>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GL/freeglut.h"
#include "GL/gl.h"
#include "glm/glm.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "mockShader.hpp"
#include "../include/display.hpp"
#include "../include/shader.hpp"
#include "../include/shape.hpp"

TEST(test_display, constructor) {
  IShaderFactory* shaderFactory = new MockShaderFactory();
  IShape* box = new Box();

  Display display(shaderFactory, box);
  delete box;
  delete shaderFactory;
  EXPECT_EQ(1, 1);
}