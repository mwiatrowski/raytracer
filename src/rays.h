#pragma once

#include <opencv2/opencv.hpp>

#include "geometry.h"

struct Ray {
  Point origin;
  Vector direction; // TODO: Normal?
};

cv::Vec3f castRay(Ray ray, int bouncesLeft);
