#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

constexpr auto WINDOW_SIZE = 1200;

class WindowWithBuffer {
  std::string name_;
  cv::Mat frame_;

public:
  WindowWithBuffer(std::string name) : name_(std::move(name)) {
    frame_ = cv::Mat::zeros(cv::Size{WINDOW_SIZE, WINDOW_SIZE}, CV_8UC3);
    cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);
  }

  ~WindowWithBuffer() { cv::destroyWindow(name_); }

  int showAndGetKey() {
    cv::imshow(name_, frame_);
    return cv::waitKey(20);
  }

  cv::Mat &frame() { return frame_; }
};

using Point = Eigen::Vector3f;
using Vector = Eigen::Vector3f;

struct Ray {
  Point origin;
  Vector direction; // TODO: Normal?
};

class Normal {
  Eigen::Vector3f vec_;

public:
  explicit Normal(const Eigen::Vector3f &vec) { vec_ = vec.normalized(); }
  const Vector &get() const { return vec_; }
};

Vector project(const Vector &v, const Normal &n) {
  return v.dot(n.get()) * n.get();
}

Vector reflect(const Vector &v, const Normal &n) {
  return 2 * project(v, n) - v;
}

Ray bounceRay(const Ray &ray, const Point &location, const Normal &normal) {
  auto newDirection = reflect(-ray.direction, normal);
  return Ray{.origin = location + 0.001 * newDirection.normalized(),
             .direction = std::move(newDirection)};
}

struct Hit {
  Point location;
  Normal normal;
  float distance;
};

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

struct Plane {
  Point origin;
  Normal normal;
};

std::optional<Hit> getHit(const Ray &ray, const Plane &plane) {
  const auto &oR = ray.origin;
  const auto &dR = ray.direction;
  const auto &oP = plane.origin;
  const auto &nP = plane.normal;

  const auto denom = dR.dot(nP.get());
  if (denom == 0.0) {
    return {};
  }

  const auto t = (oP - oR).dot(nP.get()) / denom;
  if (t <= 0.0) {
    return {};
  }

  return Hit{
      .location = oR + t * dR, .normal = nP, .distance = (t * dR).norm()};
}

std::optional<Hit> getHit(const Ray &ray,
                          const std::tuple<Point, float> &sphere) {
  const auto &[oS, rS] = sphere; // Origin and radius of the sphere
  const auto &[oR, dR] = ray;    // Origin and direction of the ray

  const auto dO = oR - oS;

  const auto a_2 = dR.dot(dR);
  const auto a_1 = 2 * dR.dot(dO);
  const auto a_0 = dO.dot(dO) - (rS * rS);

  const auto solution = positiveSolutionsForQuadratic(a_2, a_1, a_0);
  if (solution.empty()) {
    return {};
  }

  const auto t = *std::min_element(solution.begin(), solution.end());
  auto location = Point{oR + t * dR};
  auto normal = Normal(location - oS);
  return Hit{std::move(location), std::move(normal), (t * dR).norm()};
}

const auto PLANES = std::array{Plane{{0, 0, -10}, Normal{{0, 1, 10}}}};

const auto SPHERES = std::array{std::make_tuple(Point{-0.5, 7.0, 0.0}, 1.f),
                                std::make_tuple(Point{2.5, 7.0, 0.0}, 1.f),
                                std::make_tuple(Point{0.0, 7.0, 3.0}, 1.f),
                                std::make_tuple(Point{-2.0, 6.0, -2.0}, 1.f)};

constexpr auto MAX_BOUNCES = 10;

cv::Vec3b castRay(Ray ray, int bouncesLeft = MAX_BOUNCES) {
  const auto defaultColor = cv::Vec3b{255, 255, 255};

  if (bouncesLeft <= 0) {
    return defaultColor;
  }

  auto closestHit = std::optional<Hit>{};
  auto updateClosestHit = [&closestHit](const std::optional<Hit> &hit) {
    if (hit.has_value()) {
      if (!closestHit || hit->distance < closestHit->distance) {
        closestHit = hit;
      }
    }
  };

  for (const auto &sphere : SPHERES) {
    updateClosestHit(getHit(ray, sphere));
  }
  for (const auto &plane : PLANES) {
    updateClosestHit(getHit(ray, plane));
  }

  if (closestHit.has_value()) {
    const auto &[location, normal, distance] = *closestHit;
    const auto bouncedRay = bounceRay(ray, location, normal);
    const auto bouncedRayColor = castRay(bouncedRay, bouncesLeft - 1);
    return (bouncedRayColor * 0.85);
  }

  return defaultColor;
}

void renderScene(cv::Mat &frame) {
  auto width = frame.cols;
  auto height = frame.rows;

  for (auto row = 0; row < height; ++row) {
    for (auto col = 0; col < width; ++col) {
      auto planeZ = std::lerp(0.7f, -0.7f, (row + 0.5f) / height);
      auto planeX = std::lerp(-0.7f, 0.7f, (col + 0.5f) / width);
      auto ray = Ray{{0, 0, 0}, {planeX, 1.0, planeZ}};

      frame.at<cv::Vec3b>(row, col) = castRay(ray);
    }
  }
}

int main(int args_count, char *args_vals[]) {
  (void)args_count;
  (void)args_vals;

  auto window = WindowWithBuffer("raytracer");

  while (true) {
    std::cout << "Rendering the next frame..." << std::endl;

    renderScene(window.frame());

    if (window.showAndGetKey() == 27) {
      break;
    }
  }
}
