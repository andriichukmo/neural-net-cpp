#pragma once

#include "types.h"
#include <string>

namespace MNIST_reader {

uint32_t readUint32(std::ifstream &file);
NeuralNet::Matrix load_images(std::string &filename);
NeuralNet::Matrix load_labels(std::string &filename);

} // namespace MNIST_reader
