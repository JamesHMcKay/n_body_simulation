#include "../../include/shader.hpp"

#include "gmock/gmock.h"

class MockShader : public IShader {
public:
  MockShader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr) {}

  void use() {};

  unsigned int get_id() const {return 1;};

  void setInt(const std::string &name, int value) const {};
};

class MockShaderFactory : public IShaderFactory {
public:
  IShader* factory_method(const char* vertexPath, const char* fragmentPath, const char* geometryPath) const override {
    IShader* result = new MockShader(vertexPath, fragmentPath, geometryPath);
    return result;
  }
};