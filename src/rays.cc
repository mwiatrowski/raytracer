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
  enum class Type { DIFFUSE, MIRROR, EMISSIVE };

  Type type;
  Vector color;
};

const auto PLANES =
    std::array{std::make_pair(Plane{{0, 0, -3}, Normal{{0, 1, 10}}},
                              Material{.type = Material::Type::MIRROR,
                                       .color = {0.8, 0.8, 1.0}}),
               std::make_pair(Plane{{0, 300, 0}, Normal{{0, 1, 0}}},
                              Material{.type = Material::Type::DIFFUSE,
                                       .color = {0.9, 0.9, 1.0}})};

const auto SPHERES =
    std::array{std::make_pair(Sphere{Point{-6.0, 15.0, 0.0}, 2.5f},
                              Material{.type = Material::Type::EMISSIVE,
                                       .color = {3.0, 2.9, 2.75}}),
               std::make_pair(Sphere{Point{0.5, 7.0, -2.5}, 0.3f},
                              Material{.type = Material::Type::EMISSIVE,
                                       .color = {3.0, 2.7, 2.25}}),
               std::make_pair(Sphere{Point{30.0, 60.0, 30.0}, 20.0f},
                              Material{.type = Material::Type::EMISSIVE,
                                       .color = {30, 25, 20}}),
               std::make_pair(Sphere{Point{2.5, 7.0, 0.0}, 1.3f},
                              Material{.type = Material::Type::DIFFUSE,
                                       .color = {1.0, 0.85, 0.85}}),
               std::make_pair(Sphere{Point{0.0, 7.0, 3.0}, 1.3f},
                              Material{.type = Material::Type::DIFFUSE,
                                       .color = {0.85, 1.0, 0.85}}),
               std::make_pair(Sphere{Point{-2.0, 6.0, -2.0}, 1.3f},
                              Material{.type = Material::Type::DIFFUSE,
                                       .color = {0.85, 0.85, 1.0}}),
               std::make_pair(Sphere{Point{7.0, 15.0, -7.0}, 10.f},
                              Material{.type = Material::Type::DIFFUSE,
                                       .color = {0.8, 1.0, 0.8}}),
               std::make_pair(Sphere{Point{-15.0, 30.0, -15.0}, 20.f},
                              Material{.type = Material::Type::DIFFUSE,
                                       .color = {0.8, 1.0, 0.8}})};

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
  const auto orientedNormal =
      (normal.get().dot(ray.direction) <= 0) ? normal.get() : -normal.get();
  auto newDirection = (orientedNormal + randomPointOnUnitSphere()).normalized();
  return Ray{.origin = location + 0.001 * newDirection,
             .direction = std::move(newDirection)};
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

Vector castRay(Ray ray, int bouncesLeft) {
  if (bouncesLeft <= 0) {
    return {0.0, 0.0, 0.0};
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

    switch (material.type) {
    case Material::Type::DIFFUSE:
      [[fallthrough]];
    case Material::Type::MIRROR: {
      const auto bouncedRay = (material.type == Material::Type::DIFFUSE)
                                  ? bounceRayDiffuse(ray, location, normal)
                                  : bounceRayMirror(ray, location, normal);
      const auto bouncedRayColor = castRay(bouncedRay, bouncesLeft - 1);
      return bouncedRayColor.cwiseProduct(material.color);
    }
    case Material::Type::EMISSIVE: {
      return material.color;
    }
    };
  }

  return {0.05, 0.05, 0.07};
}
