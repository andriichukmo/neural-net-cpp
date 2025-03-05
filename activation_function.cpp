#pragma once

#include "activation_function.h"
#include "types.h"
#include <cassert>

namespace NeuralNet {

ActivationFunction::ActivationFunction(const Function function,
                                       const Function dfunction)
    : function_(std::move(function)), dfunction_(std::move(dfunction)) {}

Matrix ActivationFunction::Activate(const Matrix &x) const {
  assert(function_);
  return x.unaryExpr(function_);
}

Matrix ActivationFunction::ActivateDer(const Matrix &x) const {
  assert(dfunction_);
  return x.unaryExpr(dfunction_);
}

ActivationFunction ReLU() {
  static ActivationFunction ReLU([](double t) -> double { return (t > 0) * t; },
                                 [](double t) -> double { return (t > 0); });
  return ReLU;
}

ActivationFunction Sigmoid() {
  static ActivationFunction Sigmoid(
      [](double t) -> double { return 1.0 / (1.0 + std::exp(-t)); },
      [](double t) -> double {
        double tt = 1.0 / (1 + std::exp(-t));
        return tt * (1 - tt);
      });
  return Sigmoid;
}

ActivationFunction Id() {
  static ActivationFunction Id([](double t) -> double { return t; },
                               [](double t) -> double { return 1.0; });
  return Id;
}

} // namespace NeuralNet
