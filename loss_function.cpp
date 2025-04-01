#include "loss_function.h"
#include "types.h"

namespace NeuralNet {

double LossFunction::Difference(const Matrix &predictions,
                                const Matrix &targets) const {
  const double eps = 1e-12;
  int num_samples = predictions.cols();
  double total_loss = 0.0;
  for (int i = 0; i < num_samples; ++i) {
    total_loss -=
        (targets.col(i).array() * (predictions.col(i).array().max(eps)).log())
            .sum();
  }
  return total_loss / num_samples;
}

Matrix LossFunction::Gradient(const Matrix &predictions,
                              const Matrix &targets) const {
  int num_samples = predictions.cols();
  return (predictions - targets) / static_cast<double>(num_samples);
}

} // namespace NeuralNet
