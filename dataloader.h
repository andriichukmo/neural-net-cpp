#pragma once

#include "types.h"

namespace NeuralNet {

struct Batch {
  Matrix x;
  Matrix target;
};

class DataLoader {
public:
  DataLoader(Matrix &X, Matrix &T);
  void shuffle();
  std::vector<Batch> batches(int batch_size) const;

private:
  Matrix X_;
  Matrix T_;
};

} // namespace NeuralNet
