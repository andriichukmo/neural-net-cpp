#include "nnet.h"
#include "layer.h"
#include "loss_function.h"
#include "types.h"
#include <cassert>
#include <iostream>

namespace NeuralNet {

void NeuralNetwork::AddLayer(Layer &&layer) {
  layers_.push_back(std::move(layer));
}

Matrix NeuralNetwork::Predict(const Matrix &x) const {
  assert(!layers_.empty());

  Matrix res = x;

  for (auto &layer : layers_) {
    res = layer.Predict(res);
  }

  return res;
}

double NeuralNetwork::Train(const DataLoader &data, BatchSize batch_size,
                            double learn_rate,
                            const LossFunction &loss_function, Epoch epochs) {
  assert(epochs > 0);
  double last_epoch_loss = 0.0;
  class ModeGuard {
  public:
    ModeGuard(NeuralNetwork &n) : net(n) { net.SetLearningMode(); }
    ~ModeGuard() { net.SetReleaseMode(); }

  private:
    NeuralNetwork &net;
  } guard(*this);
  for (int e = 0; e < epochs; ++e) {
    last_epoch_loss =
        TrainOneEpoch(data, batch_size, learn_rate, loss_function);
    std::cout << "During Epoch " << e + 1 << " gained Loss: " << last_epoch_loss
              << std::endl;
  }
  return last_epoch_loss;
}

void NeuralNetwork::SetLearningMode() {
  std::cout << "Setting Net to Learning Mode ...\n";
  for (auto &layer : layers_) {
    layer.switchToMode(Layer::Mode::Learning);
  }
}

void NeuralNetwork::SetReleaseMode() {
  std::cout << "Setting Net to Release Mode ...\n";
  for (auto &layer : layers_) {
    layer.switchToMode(Layer::Mode::Release);
  }
}

double NeuralNetwork::TrainOneEpoch(const DataLoader &data,
                                    BatchSize batch_size, double learn_rate,
                                    const LossFunction &loss_function) {
  DataLoader data_copied = data;
  data_copied.shuffle();
  double epoch_loss = 0.0;
  for (const Batch &batch : data_copied.batches(batch_size)) {
    Matrix output = Forward(batch.x);
    double loss_val = loss_function.Difference(output, batch.target);
    Matrix grad = loss_function.Gradient(output, batch.target);
    Backward(grad, learn_rate);
    epoch_loss += loss_val;
  }

  int number_of_batches = data_copied.numBatches(batch_size);
  epoch_loss /= number_of_batches;
  return epoch_loss;
}

Matrix NeuralNetwork::Forward(const Matrix &x) {
  Matrix res = x;
  for (auto &layer : layers_) {
    res = layer.Forward(res);
  }
  return res;
}

void NeuralNetwork::Backward(const Matrix &grad, double learn_rate) {
  Matrix cur_grad = grad;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    cur_grad = it->Backward(cur_grad, learn_rate);
  }
}

} // namespace NeuralNet
