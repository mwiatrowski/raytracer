cmake_minimum_required(VERSION 3.1)
project (raytracer CXX)

set(CMAKE_CXX_STANDARD 20)

# add_compile_options(-g)
add_compile_options(-O3)

add_compile_options(-Wall -Wextra -pedantic -Werror)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

set(RAYTRACER_EXECUTABLE "raytracer")
add_executable(${RAYTRACER_EXECUTABLE}
    src/geometry.cc
    src/main.cc
    src/math.cc
    src/rays.cc
    src/window.cc
)

find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${RAYTRACER_EXECUTABLE} ${OpenCV_LIBS})

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(${RAYTRACER_EXECUTABLE} Eigen3::Eigen)
