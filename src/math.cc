#include "math.h"

#include <cmath>

std::vector<float> positiveSolutionsForQuadratic(float a_2, float a_1,
                                                 float a_0) {
  if (a_2 == 0.0) {
    // TODO: Solve the linear equation instead.
    return {};
  }

  const auto delta = a_1 * a_1 - 4 * a_2 * a_0;
  if (delta < 0) {
    return {};
  }
  const auto deltaSqrt = std::sqrt(delta);

  auto solutions = std::vector<float>{};
  if (auto x1 = (-a_1 - deltaSqrt) / (2 * a_2); x1 >= 0.0) {
    solutions.push_back(x1);
  }
  if (auto x2 = (-a_1 + deltaSqrt) / (2 * a_2); x2 >= 0.0) {
    solutions.push_back(x2);
  }
  return solutions;
}
