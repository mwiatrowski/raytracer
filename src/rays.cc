#include "rays.h"

#include <array>
#include <optional>
#include <random>

#include "geometry.h"
#include "math.h"

namespace {

struct Plane {
  Point origin;
  Normal normal;
};

struct Sphere {
  Point origin;
  float radius;
};

struct Material {
  bool diffuse;
};

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

Ray bounceRayMirror(const Ray &ray, const Point &location,
                    const Normal &normal) {
  auto newDirection = reflect(-ray.direction, normal);
  return Ray{.origin = location + 0.001 * newDirection.normalized(),
             .direction = std::move(newDirection)};
}

Ray bounceRayDiffuse(const Ray &ray, const Point &location,
                     const Normal &normal) {
  (void)ray;
  auto newDirection = (normal.get() + randomPointOnUnitSphere()).normalized();
  return Ray{.origin = location + 0.001 * newDirection,
             .direction = std::move(newDirection)};
}

Ray bounceRay(const Ray &ray, const Point &location, const Normal &normal,
              bool diffuse) {
  return diffuse ? bounceRayDiffuse(ray, location, normal)
                 : bounceRayMirror(ray, location, normal);
}

struct Hit {
  Point location;
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

} // namespace

cv::Vec3f castRay(Ray ray, int bouncesLeft) {
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
