#include <array>
#include <iostream>
#include <thread>
#include <variant>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rays.h"
#include "synchronized_queue.h"
#include "window.h"

constexpr auto WINDOW_SIZE = 800;
constexpr auto MAX_RAY_BOUNCES = 10;

void renderScene(cv::Mat &frameBuffer, cv::Vec2f subpixelDisplacement) {
  auto width = frameBuffer.cols;
  auto height = frameBuffer.rows;

  for (auto row = 0; row < height; ++row) {
    for (auto col = 0; col < width; ++col) {
      auto planeZ = std::lerp(0.7f, -0.7f,
                              (row + 0.5f + subpixelDisplacement[0]) / height);
      auto planeX = std::lerp(-0.7f, 0.7f,
                              (col + 0.5f + subpixelDisplacement[1]) / width);
      auto ray = Ray{{0, 0, 0}, {planeX, 1.0, planeZ}};
      const auto color = castRay(ray, MAX_RAY_BOUNCES);
      frameBuffer.at<cv::Vec3f>(row, col) =
          cv::Vec3f{color[0], color[1], color[2]};
    }
  }
}

struct RenderingTask {
  cv::Mat frameBuffer;
  cv::Vec2f subpixelDisplacement;
};

struct StopTask {};

using Task = std::variant<RenderingTask, StopTask>;

void raytracerLoop(SynchronizedQueue<Task> &source,
                   SynchronizedQueue<cv::Mat> &sink) {
  while (true) {
    auto task = source.popBlocking();

    if (std::holds_alternative<StopTask>(task)) {
      std::cout << "Thread " << std::this_thread::get_id() << " is quitting"
                << std::endl;
      break;
    }

    auto renderingTask = std::move(std::get<RenderingTask>(task));

    std::cout << "Thread " << std::this_thread::get_id()
              << " is rendering the next frame" << std::endl;

    renderScene(renderingTask.frameBuffer, renderingTask.subpixelDisplacement);
    sink.push(std::move(renderingTask.frameBuffer));
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

class SupersamplingPatternGenerator {
  int index_ = 0;
  const std::vector<cv::Vec2f> pattern_ = {
      cv::Vec2f{0.25, 0.25}, cv::Vec2f{0.25, -0.25}, cv::Vec2f{-0.25, 0.25},
      cv::Vec2f{-0.25, -0.25}};

public:
  cv::Vec2f get() {
    auto result = pattern_.at(index_);
    index_ = (index_ + 1) % pattern_.size();
    return result;
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto aaSampleGenerator = SupersamplingPatternGenerator{};

  auto workerThreads = std::vector<std::thread>{};
  auto workerTasks = SynchronizedQueue<Task>{};
  auto renderedFrames = SynchronizedQueue<cv::Mat>{};

  const auto numWorkerThreads = std::thread::hardware_concurrency();
  for (size_t i = 1; i <= numWorkerThreads; ++i) {
    cv::Mat frameBuffer =
        cv::Mat::zeros(cv::Size{WINDOW_SIZE, WINDOW_SIZE}, CV_32FC3);
    workerTasks.push(
        RenderingTask{std::move(frameBuffer), aaSampleGenerator.get()});

    workerThreads.emplace_back(
        [&] { raytracerLoop(workerTasks, renderedFrames); });
  }

  {
    cv::Mat renderTarget =
        cv::Mat::zeros(cv::Size{WINDOW_SIZE, WINDOW_SIZE}, CV_8UC3);
    auto window = Window("raytracer");
    auto accumulator = ImageAccumulator{};

    while (true) {
      while (const auto frameBuffer = renderedFrames.tryPop()) {
        std::cout << "UI thread got a new image" << std::endl;
        accumulator.accumulate(*frameBuffer);

        workerTasks.push(
            RenderingTask{std::move(*frameBuffer), aaSampleGenerator.get()});
      }

      accumulator.convertToIntegral(renderTarget);
      if (window.showAndGetKey(renderTarget) == 27) {
        std::cout << "Exiting the UI loop" << std::endl;
        break;
      }
    }
  }

  workerTasks.flush();
  for ([[maybe_unused]] const auto &worker : workerThreads) {
    std::cout << "Sending a stop request to a worker thread" << std::endl;
    workerTasks.push(StopTask{});
  }
  for (auto &worker : workerThreads) {
    worker.join();
  }
}
