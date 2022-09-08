#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

template <typename T> class SynchronizedQueue {
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable waitable_;

public:
  void push(T val) {
    auto lock = std::unique_lock{mutex_};
    queue_.push(std::move(val));
    waitable_.notify_one();
  }

  T popBlocking() {
    auto lock = std::unique_lock{mutex_};
    waitable_.wait(lock, [this] { return !queue_.empty(); });
    auto value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  std::optional<T> tryPop() {
    auto lock = std::unique_lock{mutex_};
    if (queue_.empty()) {
      return {};
    }
    auto value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  void flush() {
    auto lock = std::unique_lock{mutex_};
    auto emptyQueue = std::queue<T>{};
    std::swap(queue_, emptyQueue);
  }
};
