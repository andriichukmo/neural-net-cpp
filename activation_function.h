#pragma once

#include "types.h"

namespace NeuralNet {

class ActivationFunction {
  using Signature = double(double);
  using Function = std::function<Signature>;

public:
  ActivationFunction(Function function, Function dfunction);
  Matrix Apply(const Matrix &x) const;
  Matrix Differential(const Matrix &x) const;

private:
  Function function_;
  Function dfunction_;
};

ActivationFunction ReLU();
ActivationFunction Sigmoid();
ActivationFunction Id();

} // namespace NeuralNet
