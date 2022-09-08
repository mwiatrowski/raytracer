#include "geometry.h"

Vector project(const Vector &v, const Normal &n) {
  return v.dot(n.get()) * n.get();
}

Vector reflect(const Vector &v, const Normal &n) {
  return 2 * project(v, n) - v;
}
