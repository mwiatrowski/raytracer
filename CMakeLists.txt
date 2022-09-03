cmake_minimum_required(VERSION 3.1)
project (raytracer CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -pedantic -Werror)

set(RAYTRACER_EXECUTABLE "raytracer")
add_executable(${RAYTRACER_EXECUTABLE}
    src/raytracer.cc
)

find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${RAYTRACER_EXECUTABLE} ${OpenCV_LIBS})
