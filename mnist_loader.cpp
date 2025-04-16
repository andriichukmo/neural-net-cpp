#include "mnist_loader.h"
#include "types.h"
#include <cstdint>
#include <fstream>
#include <netinet/in.h>
#include <stdexcept>

namespace MNIST_reader {

using namespace NeuralNet;

uint32_t readUint32(std::ifstream &file) {
  uint32_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return ntohl(value);
}

Matrix load_images(std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Can't open file " + filename);
  }

  uint32_t number_for_check = readUint32(file);
  if (number_for_check != 2051) {
    throw std::runtime_error(filename + " is corrupted");
  }
  // magic number in MNIST file for check is it corrupted

  uint32_t number_of_images = readUint32(file);
  uint32_t number_of_rows = readUint32(file);
  uint32_t number_of_columns = readUint32(file);
  const uint32_t image_size = number_of_rows * number_of_columns;
  Matrix images(image_size, number_of_images);

  for (uint32_t i = 0; i < number_of_images; ++i) {
    std::vector<unsigned char> buffer(image_size);
    file.read(reinterpret_cast<char *>(buffer.data()), image_size);
    for (uint32_t j = 0; j < image_size; ++j) {
      images(j, i) = static_cast<double>(buffer[j]) / 255.0;
    }
  }

  return images;
}

Matrix load_labels(std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Can't open file " + filename);
  }

  uint32_t number_for_check = readUint32(file);
  if (number_for_check != 2049) {
    throw std::runtime_error(filename + " is corrupted");
  }
  uint32_t number_of_labels = readUint32(file);

  int number_of_digits = 10;
  Matrix res = Matrix::Zero(number_of_digits, number_of_labels);

  for (uint32_t i = 0; i < number_of_labels; ++i) {
    unsigned char label = 0;
    file.read(reinterpret_cast<char *>(&label), sizeof(label));
    if (label > number_of_digits) {
      throw std::runtime_error("Label out of range in file " + filename);
    }
    res(label, i) = 1.0;
  }
  return res;
}

} // namespace MNIST_reader
