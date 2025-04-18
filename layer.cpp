#include "layer.h"
#include "activation_function.h"
#include "types.h"
#include <cassert>
#include <memory>

namespace NeuralNet {

Layer::Layer(In in_size, Out out_size, ActivationFunction &&activation_function)
    : A_(Matrix::Random(out_size, in_size)), b_(Vector::Random(out_size)),
      activation_function_(std::move(activation_function)), cache_(nullptr) {}

void Layer::switchToMode(Mode mode) {
  if (mode == Mode::Learning) {
    if (!cache_) {
      cache_ = std::make_unique<Cache>();
    }
  } else {
    cache_.reset();
  }
}

Matrix Layer::Forward(const Matrix &x) {
  assert(x.rows() == A_.cols());
  assert(cache_);
  Matrix y = (A_ * x).colwise() + b_;
  cache_->x = x;
  cache_->y = y;
  return activation_function_.Apply(y);
}

Matrix Layer::Predict(const Matrix &x) const {
  assert(x.rows() == A_.cols());
  Matrix y = (A_ * x).colwise() + b_;
  return activation_function_.Apply(y);
}

Matrix Layer::Backward(const Matrix &grad, double learn_rate) {
  assert(learn_rate >= 0);
  assert(grad.rows() == A_.rows());
  assert(cache_);

  Matrix dz =
      grad.array() * activation_function_.Differential(cache_->y).array();
  Matrix grad_A = dz * cache_->x.transpose();
  Matrix grad_b = dz.rowwise().sum();
  A_ -= learn_rate * grad_A;
  b_ -= learn_rate * grad_b;

  return A_.transpose() * dz;
}

} // namespace NeuralNet
