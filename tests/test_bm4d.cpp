#include <gtest/gtest.h>
#include <bm4d-gpu/kernels.cuh>

#include <iostream>

TEST(TestDistanceComputation, ComputeSamePatch)
{
    // Arrange
    std::vector<uchar> image = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const uint3 image_size = make_uint3(3, 3, 1);
    const int3 reference_patch = make_int3(0, 0, 0);
    const int3 compare_patch = make_int3(0, 0, 0);
    const int patch_size = 3;

    // Act
    const float distance = dist(image.data(), image_size, reference_patch, compare_patch, patch_size);

    // Assert
    EXPECT_EQ(0.f, distance);
}

TEST(TestDistanceComputation, ComputeDifferentPatchWithBoundaryCondition)
{
    // Arrange
    std::vector<uchar> image = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const uint3 image_size = make_uint3(3, 3, 1);
    const int3 reference_patch = make_int3(0, 0, 0);
    const int3 compare_patch = make_int3(1, 1, 0);
    const int patch_size = 3;

    // Act
    const float distance = dist(image.data(), image_size, reference_patch, compare_patch, patch_size);

    // Assert
    ASSERT_NEAR(84. / 9., distance, 0.0001);
}
