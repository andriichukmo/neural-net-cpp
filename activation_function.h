#pragma once

#include "types.h"

namespace NeuralNet {

class ActivationFunction {
public:
  ActivationFunction(const Function function, const Function dfunction);
  Matrix Activate(const Matrix &x) const;
  Matrix Derivative(const Matrix &x) const;

private:
  Function function_;
  Function dfunction_;
};

ActivationFunction ReLU();
ActivationFunction Sigmoid();
ActivationFunction Id();

} // namespace NeuralNet
