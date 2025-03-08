#pragma once

#include "nnet.h"

namespace NeuralNet {

void NeuralNetwork::AddLayer(const LinearLayer &layer) {
  layers_.push_back(layer);
}

Matrix NeuralNetwork::Predict(const Matrix &x) {
  assert(!layers_.empty());

  Matrix res = x;

  for (auto &layer : layers_) {
    res = layer.Forward(res);
  }

  return res;
}

double NeuralNetwork::Train(const Matrix &x, const Matrix &targets,
                            double learn_rate,
                            const LossFunction &loss_function) {
  assert(!layers_.empty());
  assert(learn_rate >= 0);

  Matrix res = x;
  for (auto &layer : layers_) {
    res = layer.Forward(res);
  }

  assert(res.rows() == targets.rows() && res.cols() == targets.cols());

  double loss_value = loss_function.Difference(res, targets);

  Matrix grad = loss_function.Gradient(res, targets);

  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    grad = it->Backward(grad, learn_rate);
  }

  return loss_value;
}

double NeuralNetwork::TrainEpochs(const Matrix &x, const Matrix &targets,
                                  double learn_rate,
                                  const LossFunction &loss_function,
                                  int epochs) {
  assert(epochs > 0);
  double loss_value = 0.0;
  for (int i = 0; i < epochs; ++i) {
    loss_value = Train(x, targets, learn_rate, loss_function);
  }
  return loss_value;
}

} // namespace NeuralNet
