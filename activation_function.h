#pragma once

#include "types.h"

namespace NeuralNet {

class ActivationFunction {
public:
  virtual Matrix Apply(const Matrix &x) const = 0;
  virtual Matrix Differential(const Matrix &x) const = 0;
};

class OneElementActivationFunction : public ActivationFunction {
  using Signature = double(double);
  using Function = std::function<Signature>;

public:
  OneElementActivationFunction(Function function, Function dfunction);
  Matrix Apply(const Matrix &x) const;
  Matrix Differential(const Matrix &x) const;

private:
  Function function_;
  Function dfunction_;
};

namespace EasyActivationFunc {

OneElementActivationFunction ReLU();
OneElementActivationFunction Sigmoid();
OneElementActivationFunction Id();

} // namespace EasyActivationFunc

class SoftMaxActivation : public ActivationFunction {
public:
  Matrix Apply(const Matrix &x) const;
  Matrix Differential(const Matrix &x) const;
};

SoftMaxActivation SoftMax();

} // namespace NeuralNet
