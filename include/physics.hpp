#ifndef physics_hpp
#define physics_hpp

#include "vector_types.h"
#include "consts.hpp"

class Physics {
public:

  __device__ __host__
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
  };
};

#endif