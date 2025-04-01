#pragma once

#include "activation_function.h"
#include "types.h"

namespace NeuralNet {

OneElementActivationFunction::OneElementActivationFunction(Function function,
                                                           Function dfunction)
    : function_(std::move(function)), dfunction_(std::move(dfunction)) {}

Matrix OneElementActivationFunction::Apply(const Matrix &x) const {
  assert(function_);
  return x.unaryExpr(function_);
}

Matrix OneElementActivationFunction::Differential(const Matrix &x) const {
  assert(dfunction_);
  return x.unaryExpr(dfunction_);
}

OneElementActivationFunction ReLU() {
  static OneElementActivationFunction ReLU(
      [](double t) -> double { return (t > 0) * t; },
      [](double t) -> double { return (t > 0); });
  return ReLU;
}

OneElementActivationFunction Sigmoid() {
  static OneElementActivationFunction Sigmoid(
      [](double t) -> double { return 1.0 / (1.0 + std::exp(-t)); },
      [](double t) -> double {
        double tt = 1.0 / (1 + std::exp(-t));
        return tt * (1 - tt);
      });
  return Sigmoid;
}

OneElementActivationFunction Id() {
  static OneElementActivationFunction Id(
      [](double t) -> double { return t; },
      [](double t) -> double { return 1.0; });
  return Id;
}

Matrix SoftMaxActivation::Apply(const Matrix &x) const {
  Matrix res(x.rows(), x.cols());
  for (int i = 0; i < x.cols(); ++i) {
    Vector col = x.col(i);
    Vector exps = (col.array() - col.maxCoeff()).exp();
    res.col(i) = exps / exps.sum();
  }
  return res;
}

Matrix SoftMaxActivation::Differential(const Matrix &x) const {
  return Apply(x);
}

SoftMaxActivation SoftMax() { return SoftMaxActivation(); }

} // namespace NeuralNet
