#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class WindowWithBuffer {
public:
  WindowWithBuffer(std::string name) : name_(std::move(name)) {
    frame_ = cv::Mat::zeros(cv::Size{512, 512}, CV_8UC3);
    cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);
  }

  ~WindowWithBuffer() { cv::destroyWindow(name_); }

  int showAndGetKey() {
    cv::imshow(name_, frame_);
    return cv::waitKey(1);
  }

  cv::Mat &frame() { return frame_; }

private:
  std::string name_;
  cv::Mat frame_;
};

int main(int args_count, char *args_vals[]) {
  (void)args_count;
  (void)args_vals;

  auto window = WindowWithBuffer("NICE_WINDOW");

  while (true) {
    // Render the next frame
    // ...

    if (window.showAndGetKey() == 27) {
      break;
    }
  }
}
