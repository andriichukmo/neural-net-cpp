#include "loss_function.h"
#include "types.h"

namespace NeuralNet {

double LossFunction::Difference(const Matrix &predictions,
                                const Matrix &targets) const {
  auto diff = predictions - targets;
  return diff.array().square().mean();
}

Matrix LossFunction::Gradient(const Matrix &predictions,
                              const Matrix &targets) const {
  return (2.0 / static_cast<double>(predictions.rows())) *
         (predictions - targets);
}

} // namespace NeuralNet
