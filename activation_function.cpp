#include "activation_function.h"
#include "types.h"
#include <cassert>

namespace NeuralNet {

ActivationFunction::ActivationFunction(Function function, Function dfunction)
    : function_(std::move(function)), dfunction_(std::move(dfunction)) {}

Matrix ActivationFunction::Apply(const Matrix &x) const {
  assert(function_);
  return function_(x);
}

Matrix ActivationFunction::Differential(const Matrix &x) const {
  assert(dfunction_);
  return dfunction_(x);
}

ActivationFunction ActivationFuncs::ReLU() {
  ActivationFunction ReLU(
      [](const Matrix &x) -> Matrix { return x.array().max(0.0).matrix(); },
      [](const Matrix &x) -> Matrix {
        return (x.array() > 0.0).cast<double>().matrix();
      });
  return ReLU;
}

ActivationFunction ActivationFuncs::Sigmoid() {
  ActivationFunction Sigmoid(
      [](const Matrix &x) -> Matrix {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
      },
      [](const Matrix &x) -> Matrix {
        Matrix y = (1.0 / (1.0 + (-x.array()).exp())).matrix();
        return (y.array() * (1.0 - y.array())).matrix();
      });
  return Sigmoid;
}

ActivationFunction ActivationFuncs::Id() {
  ActivationFunction Id([](const Matrix &x) -> Matrix { return x; },
                        [](const Matrix &x) -> Matrix {
                          return Matrix::Ones(x.rows(), x.cols());
                        });
  return Id;
}

ActivationFunction ActivationFuncs::SoftMax() {
  ActivationFunction SoftMax(
      [](const Matrix &x) -> Matrix {
        Matrix res(x.rows(), x.cols());
        for (int i = 0; i < x.cols(); ++i) {
          Vector col = x.col(i);
          Vector exps = (col.array() - col.maxCoeff()).exp();
          res.col(i) = exps / exps.sum();
        }
        return res;
      },
      [](const Matrix &x) -> Matrix {
        return Matrix::Ones(x.rows(), x.cols());
      });
  return SoftMax;
}

} // namespace NeuralNet
