#pragma once

#include <string>

#include <opencv2/opencv.hpp>

class Window {
  std::string name_;

public:
  Window(std::string name);
  ~Window();
  int showAndGetKey(cv::Mat &frame);
};
