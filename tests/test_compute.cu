#include <iostream>

#include "gtest/gtest.h"
#include "../include/kernel.cuh"
#include "vector_types.h"

TEST(test_compute, add_acceleration) {
    float4 p1 = make_float4(0, 0, 0, 0);
    float4 p2 = make_float4(0, 0, 0, 0);
    float3 a = make_float3(0, 0, 0);
    float3 result = add_acceleration(p1, p2, a);

    EXPECT_EQ(add_acceleration(p1, p2, a).x, make_float3(0, 0, 0).x);
}