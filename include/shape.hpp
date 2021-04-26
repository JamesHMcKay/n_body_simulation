#ifndef SHAPE_H
#define SHAPE_H
#include <iostream>
#include <vector>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GL/freeglut.h"
#include "GL/gl.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.hpp"
#include "consts.hpp"

class IShape {
public:
  virtual void create_frame_buffers() = 0;

  virtual void shut_down() = 0;

  virtual void bind_vertex_arrays() const = 0;

  virtual ~IShape() {}
};

class Box : public IShape {
public:
  Box() {}

  void create_frame_buffers();

  void shut_down() override {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
  };

  void bind_vertex_arrays() const override {
    glBindVertexArray(VAO);
  };

private:

  unsigned int VAO;
  unsigned int VBO;

};

// factory method not currently used
class IShapeFactory {
  public:
  virtual ~IShapeFactory(){};
  virtual IShape* factory_method() const = 0;

  IShape* get_shape() const {
    IShape* shape = this->factory_method();
    return shape;
  }
};

class BoxFactory : public IShapeFactory {
  public:
  IShape* factory_method() const override {
    IShape* result = new Box();
    return result;
  }
};



#endif