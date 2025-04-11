#pragma once

#include "activation_function.h"
#include "types.h"
#include <memory>

namespace NeuralNet {
enum In : Index;
enum Out : Index;
class Layer {
public:
  struct Cache {
    Matrix x;
    Matrix y;
  };
  enum class Mode { Release, Learning };

  Layer(In in_size, Out out_size, ActivationFunction &&activation_function);
  void switchToMode(Mode mode);
  Matrix Forward(const Matrix &x);
  Matrix Backward(const Matrix &grad, double learn_rate);

private:
  Matrix A_;
  Vector b_;
  ActivationFunction activation_function_;

  std::unique_ptr<Cache> cache_;
};

} // namespace NeuralNet
