#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "rays.h"
#include "synchronized_queue.h"
#include "window.h"

constexpr auto WINDOW_SIZE = 800;
constexpr auto MAX_RAY_BOUNCES = 10;

void renderScene(cv::Mat &frameBuffer) {
  auto width = frameBuffer.cols;
  auto height = frameBuffer.rows;

  for (auto row = 0; row < height; ++row) {
    for (auto col = 0; col < width; ++col) {
      auto planeZ = std::lerp(0.7f, -0.7f, (row + 0.5f) / height);
      auto planeX = std::lerp(-0.7f, 0.7f, (col + 0.5f) / width);
      auto ray = Ray{{0, 0, 0}, {planeX, 1.0, planeZ}};
      frameBuffer.at<cv::Vec3f>(row, col) = castRay(ray, MAX_RAY_BOUNCES);
    }
  }
}

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
