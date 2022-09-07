#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <tuple>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

constexpr auto WINDOW_SIZE = 800;

class Window {
  std::string name_;

public:
  Window(std::string name) : name_(std::move(name)) {
    cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);
  }

  ~Window() { cv::destroyWindow(name_); }

  int showAndGetKey(cv::Mat &frame) {
    cv::imshow(name_, frame);
    return cv::waitKey(10);
  }
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

Ray bounceRayMirror(const Ray &ray, const Point &location,
                    const Normal &normal) {
  auto newDirection = reflect(-ray.direction, normal);
  return Ray{.origin = location + 0.001 * newDirection.normalized(),
             .direction = std::move(newDirection)};
}

Vector randomPointOnUnitSphere() {
  static auto randomDevice = std::random_device{};
  static auto generator = std::mt19937{randomDevice()};
  static auto uniform = std::uniform_real_distribution<>(-1.0, 1.0);

  const auto rng = []() { return static_cast<float>(uniform(generator)); };

  auto vec = Vector{};
  do {
    vec = Vector{rng(), rng(), rng()};
  } while (vec.norm() > 1.0);
  return vec;
}

Ray bounceRayDiffuse(const Ray &ray, const Point &location,
                     const Normal &normal) {
  (void)ray;
  auto newDirection = (normal.get() + randomPointOnUnitSphere()).normalized();
  return Ray{.origin = location + 0.001 * newDirection,
             .direction = std::move(newDirection)};
}

struct Material {
  bool diffuse;
};

Ray bounceRay(const Ray &ray, const Point &location, const Normal &normal,
              bool diffuse) {
  return diffuse ? bounceRayDiffuse(ray, location, normal)
                 : bounceRayMirror(ray, location, normal);
}

struct Hit {
  Point location;
  Normal normal;
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

  return Hit{.location = oR + t * dR, .normal = nP};
}

struct Sphere {
  Point origin;
  float radius;
};

std::optional<Hit> getHit(const Ray &ray, const Sphere &sphere) {
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
  return Hit{std::move(location), std::move(normal)};
}

constexpr auto MATERIAL_DIFFUSE = Material{.diffuse = true};
constexpr auto MATERIAL_MIRROR = Material{.diffuse = false};

const auto PLANES = std::array{
    std::make_pair(Plane{{0, 0, -3}, Normal{{0, 1, 10}}}, MATERIAL_DIFFUSE)};

const auto SPHERES = std::array{
    std::make_pair(Sphere{Point{-0.5, 7.0, 0.0}, 1.3f}, MATERIAL_DIFFUSE),
    std::make_pair(Sphere{Point{2.5, 7.0, 0.0}, 1.3f}, MATERIAL_MIRROR),
    std::make_pair(Sphere{Point{0.0, 7.0, 3.0}, 1.3f}, MATERIAL_MIRROR),
    std::make_pair(Sphere{Point{-2.0, 6.0, -2.0}, 1.3f}, MATERIAL_MIRROR),
    std::make_pair(Sphere{Point{7.0, 15.0, -7.0}, 10.f}, MATERIAL_DIFFUSE),
    std::make_pair(Sphere{Point{-15.0, 30.0, -15.0}, 20.f}, MATERIAL_DIFFUSE)};

constexpr auto MAX_BOUNCES = 10;

cv::Vec3f castRay(Ray ray, int bouncesLeft = MAX_BOUNCES) {
  const auto defaultColor = cv::Vec3f{1.0, 1.0, 1.0};

  if (bouncesLeft <= 0) {
    return defaultColor;
  }

  struct HitExt {
    Hit hit;
    float distance;
    Material material;
  };
  auto closestHit = std::optional<HitExt>{};
  auto updateClosestHit = [&closestHit](const HitExt &hitInfo) {
    if (!closestHit || hitInfo.distance < closestHit->distance) {
      closestHit = hitInfo;
    }
  };

  for (const auto &[sphere, material] : SPHERES) {
    if (const auto hit = getHit(ray, sphere)) {
      auto hitInfo = HitExt{.hit = *hit,
                            .distance = (ray.origin - hit->location).norm(),
                            .material = material};
      updateClosestHit(hitInfo);
    }
  }
  for (const auto &[plane, material] : PLANES) {
    if (const auto hit = getHit(ray, plane)) {
      auto hitInfo = HitExt{.hit = *hit,
                            .distance = (ray.origin - hit->location).norm(),
                            .material = material};
      updateClosestHit(hitInfo);
    }
  }

  if (closestHit.has_value()) {
    const auto &[hit, distance, material] = *closestHit;
    const auto &[location, normal] = hit;
    const auto bouncedRay = bounceRay(ray, location, normal, material.diffuse);
    const auto bouncedRayColor = castRay(bouncedRay, bouncesLeft - 1);
    return (bouncedRayColor * 0.85);
  }

  return defaultColor;
}

void renderScene(cv::Mat &frameBuffer) {
  auto width = frameBuffer.cols;
  auto height = frameBuffer.rows;

  for (auto row = 0; row < height; ++row) {
    for (auto col = 0; col < width; ++col) {
      auto planeZ = std::lerp(0.7f, -0.7f, (row + 0.5f) / height);
      auto planeX = std::lerp(-0.7f, 0.7f, (col + 0.5f) / width);
      auto ray = Ray{{0, 0, 0}, {planeX, 1.0, planeZ}};
      frameBuffer.at<cv::Vec3f>(row, col) = castRay(ray);
    }
  }
}

class ImageAccumulator {
  cv::Mat accumulator_;
  int samplesCount_ = 0;

public:
  void accumulate(cv::Mat const &frameBuffer) {
    accumulator_.create(frameBuffer.size(), frameBuffer.type());
    accumulator_ += frameBuffer;
    samplesCount_ += 1;
  }

  void convertToIntegral(cv::Mat &renderTarget) const {
    if (samplesCount_ == 0) {
      return;
    }
    accumulator_.convertTo(renderTarget, CV_8U, 255.0 / samplesCount_, 0.0);
  }
};

template <typename T> class SynchronizedQueue {
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable waitable_;

public:
  void push(T val) {
    auto lock = std::unique_lock{mutex_};
    queue_.push(std::move(val));
    waitable_.notify_one();
  }

  T popBlocking() {
    T value;
    {
      auto lock = std::unique_lock{mutex_};
      waitable_.wait(lock, [this] { return !queue_.empty(); });
      value = std::move(queue_.front());
      queue_.pop();
    }
    waitable_.notify_one();
    return value;
  }

  std::optional<T> tryPop() {
    auto lock = std::unique_lock{mutex_};
    if (queue_.empty()) {
      return {};
    }
    auto value = std::move(queue_.front());
    queue_.pop();
    return value;
  }
};

void raytracerLoop(SynchronizedQueue<cv::Mat> &source,
                   SynchronizedQueue<cv::Mat> &sink) {
  while (true) {
    auto frameBuffer = std::move(source.popBlocking());

    std::cout << "Thread " << std::this_thread::get_id()
              << " is rendering the next frame" << std::endl;

    renderScene(frameBuffer);
    sink.push(std::move(frameBuffer));
  }
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto workerThreads = std::vector<std::thread>{};
  auto unusedFrames = SynchronizedQueue<cv::Mat>{};
  auto renderedFrames = SynchronizedQueue<cv::Mat>{};

  const auto numWorkerThreads = std::thread::hardware_concurrency();
  for (size_t i = 1; i <= numWorkerThreads; ++i) {
    cv::Mat frameBuffer =
        cv::Mat::zeros(cv::Size{WINDOW_SIZE, WINDOW_SIZE}, CV_32FC3);
    unusedFrames.push(std::move(frameBuffer));

    workerThreads.emplace_back(
        [&] { raytracerLoop(unusedFrames, renderedFrames); });
  }

  cv::Mat renderTarget =
      cv::Mat::zeros(cv::Size{WINDOW_SIZE, WINDOW_SIZE}, CV_8UC3);
  auto window = Window("raytracer");
  auto accumulator = ImageAccumulator{};

  while (true) {
    while (const auto frameBuffer = renderedFrames.tryPop()) {
      std::cout << "UI thread got a new image" << std::endl;
      accumulator.accumulate(*frameBuffer);
      unusedFrames.push(std::move(*frameBuffer));
    }

    accumulator.convertToIntegral(renderTarget);
    if (window.showAndGetKey(renderTarget) == 27) {
      break;
    }
  }

  // TODO: Properly inform worker threads that they should exit.
}
