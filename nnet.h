#pragma once

#include "dataloader.h"
#include "layer.h"
#include "loss_function.h"
#include "types.h"

namespace NeuralNet {

enum Epoch : int;

class NeuralNetwork {
public:
  void AddLayer(Layer &&layer);

  Matrix Predict(const Matrix &x) const;

  double Train(const DataLoader &data, BatchSize batch_size, double learn_rate,
               const LossFunction &loss_function, Epoch epochs);

private:
  void SetLearningMode();
  void SetReleaseMode();
  Matrix Forward(const Matrix &x);
  void Backward(const Matrix &grad, double learn_rate);
  double TrainOneEpoch(const DataLoader &data, BatchSize batch_size,
                       double learn_rate, const LossFunction &loss_function);

  std::vector<Layer> layers_;
};

} // namespace NeuralNet
