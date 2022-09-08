#pragma once

#include <Eigen/Geometry>

using Point = Eigen::Vector3f;
using Vector = Eigen::Vector3f;

class Normal {
  Eigen::Vector3f vec_;

public:
  explicit Normal(const Eigen::Vector3f &vec) { vec_ = vec.normalized(); }
  const Vector &get() const { return vec_; }
};

Vector project(const Vector &v, const Normal &n);
Vector reflect(const Vector &v, const Normal &n);
