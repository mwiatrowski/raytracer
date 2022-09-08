#include "window.h"

Window::Window(std::string name) : name_(std::move(name)) {
  cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);
}

Window::~Window() { cv::destroyWindow(name_); }

int Window::showAndGetKey(cv::Mat &frame) {
  cv::imshow(name_, frame);
  return cv::waitKey(10);
}
