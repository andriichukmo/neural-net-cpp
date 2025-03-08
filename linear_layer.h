#pragma once

#include "activation_function.h"
#include "types.h"

namespace NeuralNet {

class LinearLayer {
public:
  LinearLayer(Index in_size, Index out_size,
              ActivationFunction activation_function);
  Matrix Forward(const Matrix &x);
  Matrix Backward(const Matrix &u, double learn_rate);

private:
  Matrix A_;
  Vector b_;
  ActivationFunction activation_function_;

  // for backward
  Matrix x_;
  Matrix z_;
  Matrix a_;
};

} // namespace NeuralNet