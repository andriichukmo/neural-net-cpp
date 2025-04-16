#include "dataloader.h"
#include "types.h"
#include <algorithm>
#include <cassert>
#include <random>

namespace NeuralNet {

BatchIterator::BatchIterator(const Matrix &X, const Matrix &T,
                             BatchSize batch_size, Pos pos)
    : X_(X), T_(T), batch_size_(batch_size), total_samples_(X.cols()),
      pos_(pos) {}

Batch BatchIterator::operator*() const {
  int cur_size = std::min(total_samples_ - pos_, batch_size_);
  Batch batch;
  batch.x = X_.block(0, pos_, X_.rows(), cur_size);
  batch.target = T_.block(0, pos_, T_.rows(), cur_size);
  return batch;
}

BatchIterator &BatchIterator::operator++() {
  pos_ += batch_size_;
  return *this;
}

bool BatchIterator::operator!=(const BatchIterator &other) const {
  return pos_ != other.pos_;
}

BatchProxy::BatchProxy(const Matrix &X, const Matrix &T, int batch_size)
    : X_(X), T_(T), batch_size_(batch_size){};

BatchIterator BatchProxy::begin() const {
  return BatchIterator(X_, T_, BatchSize{batch_size_}, Pos{0});
}

BatchIterator BatchProxy::end() const {
  return BatchIterator(X_, T_, BatchSize{batch_size_}, Pos{X_.cols()});
}

DataLoader::DataLoader(const Matrix &X, const Matrix &T) : X_(X), T_(T) {
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

BatchProxy DataLoader::batches(int batch_size) const {
  return BatchProxy(X_, T_, batch_size);
}

int DataLoader::numBatches(int batch_size) const {
  return (X_.cols() + batch_size - 1) / batch_size;
}

} // namespace NeuralNet
