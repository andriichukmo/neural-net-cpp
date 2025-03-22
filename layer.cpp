#pragma once

#include "layer.h"

namespace NeuralNet {

Layer::Layer(Index in_size, Index out_size,
             ActivationFunction activation_function)
    : A_(Matrix::Random(out_size, in_size)), b_(Vector::Random(out_size)),
      activation_function_(activation_function) {}

Matrix Layer::Forward(const Matrix &x) {
  assert(x.rows() == A_.cols());
  x_ = x;
  z_ = (A_ * x_).colwise() + b_.col(0);
  a_ = activation_function_.Apply(z_);
  return a_;
}

Matrix Layer::Backward(const Matrix &u, double learn_rate) {
  assert(learn_rate >= 0);
  assert(u.rows() == A_.rows());
  assert(u.cols() == x_.cols());

  Matrix dz = u.array() * activation_function_.Differential(z_).array();

  Matrix gradA = dz * x_.transpose();

  Matrix gradb = dz.rowwise().sum();

  A_ -= learn_rate * gradA;
  b_ -= learn_rate * gradb;

  return A_.transpose() * dz;
}

} // namespace NeuralNet
