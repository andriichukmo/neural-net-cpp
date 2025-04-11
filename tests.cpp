#include "tests.h"
#include "activation_function.h"
#include "dataloader.h"
#include "layer.h"
#include "loss_function.h"
#include "mnist_loader.h"
#include "nnet.h"
#include "types.h"
#include <iomanip>
#include <iostream>

namespace Tests {

using namespace NeuralNet;
using namespace MNIST_reader;

void test_basic() {
  constexpr Index in_size = 3;
  constexpr Index out_size = 2;
  constexpr Index samples = 4;
  constexpr Index extra_layer = 8;

  Matrix X(in_size, samples);
  X << 1, 2, 3, 4, 4, 3, 2, 1, 1, 1, 1, 1;

  Matrix T(out_size, samples);
  T << 1, 0, 1, 0, 0, 1, 0, 1;

  LossFunction loss_function;

  NeuralNetwork net;

  net.AddLayer(Layer(In{in_size}, Out{extra_layer}, ActivationFuncs::ReLU()));
  net.AddLayer(Layer(In{extra_layer}, Out{4}, ActivationFuncs::Id()));
  net.AddLayer(Layer(In{4}, Out{out_size}, ActivationFuncs::SoftMax()));
  DataLoader data(X, T);
  double learn_rate = 0.01;
  int epochs = 20;
  int batch_size = 4;

  double final_loss = net.Train(data, BatchSize{batch_size}, learn_rate,
                                loss_function, Epoch{epochs});

  std::cout << std::fixed << std::setprecision(5)
            << "Final Loss: " << final_loss << std::endl;

  Matrix predictions = net.Predict(X);

  std::cout << std::fixed << std::setprecision(5) << predictions << std::endl;
}

void test_mnist() {
  std::string train_images_path = "data/MNIST/train-images.idx3-ubyte";
  std::string train_labels_path = "data/MNIST/train-labels.idx1-ubyte";
  Matrix images = load_images(train_images_path);
  Matrix labels = load_labels(train_labels_path);
  std::cout << "Loaded " << images.cols() << " training samples." << std::endl;
  std::cout << "Each image has " << images.rows() << " pixels." << std::endl;

  LossFunction loss_function;
  NeuralNetwork net;
  constexpr int image_size = 784;
  constexpr int second_layer_size = 128;
  constexpr int third_layer_size = 64;
  constexpr int fourth_layer_size = 64;
  constexpr int number_of_digits = 10;
  net.AddLayer(
      Layer(In{image_size}, Out{second_layer_size}, ActivationFuncs::Id()));
  net.AddLayer(Layer(In{second_layer_size}, Out{third_layer_size},
                     ActivationFuncs::Sigmoid()));
  net.AddLayer(Layer(In{third_layer_size}, Out{fourth_layer_size},
                     ActivationFuncs::ReLU()));

  net.AddLayer(Layer(In{fourth_layer_size}, Out{number_of_digits},
                     ActivationFuncs::SoftMax()));

  DataLoader data(images, labels);
  double learn_rate = 1e-4;
  int num_epoch = 666;
  int batch_size = 16;
  double final_loss = net.Train(data, BatchSize{batch_size}, learn_rate,
                                loss_function, Epoch{num_epoch});
  std::cout << std::fixed << std::setprecision(5)
            << "Final Training Loss : " << final_loss << std::endl;

  Matrix predictions = net.Predict(images);
  std::cout << "Result for first 100 images :" << std::endl;
  for (int i = 0; i < 100; ++i) {
    int predicted, ans;
    predictions.col(i).maxCoeff(&predicted);
    labels.col(i).maxCoeff(&ans);
    std::cout << "My prediction for " << i + 1 << "-th image : " << predicted
              << " and right answer is " << ans << " and it's "
              << (predicted == ans ? "OK" : "BAD") << std::endl
              << predictions.col(i).transpose() << std::endl
              << labels.col(i).transpose() << std::endl;
  }
  int correct = 0;
  for (int i = 0; i < images.cols(); ++i) {
    int predicted, ans;
    predictions.col(i).maxCoeff(&predicted);
    labels.col(i).maxCoeff(&ans);
    if (ans == predicted) {
      correct++;
    }
  }
  std::cout << "And final result is " << correct << " out of " << images.cols()
            << "\n";
  std::cout << "It's about " << static_cast<double>(correct) / images.cols()
            << " rate!\n";
}

} // namespace Tests
