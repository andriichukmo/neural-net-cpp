#include "dataloader.h"
#include "types.h"
#include <cassert>
#include <random>

namespace NeuralNet {

DataLoader::DataLoader(Matrix &X, Matrix &T) : X_(X), T_(T) {
  assert(X.cols() == T.cols());
}

void DataLoader::shuffle() {
  int n = X_.cols();
  std::vector<int> ids(n);
  for (int i = 0; i < n; ++i) {
    ids[i] = i;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(ids.begin(), ids.end(), g);

  Matrix X_shuffled(X_.rows(), n);
  Matrix T_shuffled(T_.rows(), n);
  for (int i = 0; i < n; ++i) {
    X_shuffled.col(i) = X_.col(ids[i]);
    T_shuffled.col(i) = T_.col(ids[i]);
  }
  X_ = X_shuffled;
  T_ = T_shuffled;
}

std::vector<Batch> DataLoader::batches(int batch_size) const {
  std::vector<Batch> res;
  int n = X_.cols();
  for (int i = 0; i < n; i += batch_size) {
    int cur_size = std::min(batch_size, n - i);
    Batch batch;
    batch.x = X_.block(0, i, X_.rows(), cur_size);
    batch.target = T_.block(0, i, T_.rows(), cur_size);
    res.push_back(batch);
  }
  return res;
}

} // namespace NeuralNet
