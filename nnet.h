#pragma once

#include "layer.h"
#include "loss_function.h"
#include "types.h"

namespace NeuralNet {

class NeuralNetwork {
public:
  void AddLayer(Layer &&layer);

  Matrix Predict(const Matrix &x);

  double Train(const Matrix &x, const Matrix &targets, double learn_rate,
               const LossFunction &loss_function);

  double TrainEpochs(const Matrix &x, const Matrix &targets, double learn_rate,
                     const LossFunction &loss_function, int epochs);

private:
  std::vector<Layer> layers_;
};

} // namespace NeuralNet
