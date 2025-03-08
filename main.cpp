#include "activation_function.h"
#include "linear_layer.h"
#include "loss_function.h"
#include "nnet.h"
#include "types.h"
#include <iomanip>
#include <iostream>

using namespace NeuralNet;

int main() {
  const Index in_size = 3;
  const Index out_size = 2;
  const Index samples = 4;

  Matrix X(in_size, samples);
  X << 1, 2, 3, 4, 4, 3, 2, 1, 1, 1, 1, 1;

  Matrix T(out_size, samples);
  T << 1, 0, 1, 0, 0, 1, 0, 1;

  LossFunction loss_function;

  NeuralNetwork net;

  net.AddLayer(LinearLayer(in_size, 4, ReLU()));
  net.AddLayer(LinearLayer(4, out_size, Id()));

  double learn_rate = 0.01;
  int epochs = 1000;

  double final_loss = net.TrainEpochs(X, T, learn_rate, loss_function, epochs);

  std::cout << std::fixed << std::setprecision(5)
            << "Final Loss: " << final_loss << std::endl;

  Matrix predictions = net.Predict(X);

  std::cout << std::fixed << std::setprecision(5) << predictions << std::endl;
}
