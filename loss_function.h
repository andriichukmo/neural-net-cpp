#pragma once

#include "types.h"

namespace NeuralNet {

class LossFunction {
public:
  double Difference(const Matrix &predictions, const Matrix &targets) const;
  Matrix Gradient(const Matrix &predictions, const Matrix &targets) const;
};

} // namespace NeuralNet
