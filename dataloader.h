#pragma once

#include "types.h"

namespace NeuralNet {

struct Batch {
  Matrix x;
  Matrix target;
};

enum BatchSize : Index;
enum Pos : Index;

class BatchIterator {
public:
  BatchIterator(const Matrix &X, const Matrix &T, BatchSize batch_size,
                Pos pos);
  Batch operator*() const;
  BatchIterator &operator++();
  bool operator!=(const BatchIterator &other) const;

private:
  const Matrix &X_;
  const Matrix &T_;
  Index batch_size_;
  Index pos_;
  Index total_samples_;
};

class BatchProxy {
public:
  BatchProxy(const Matrix &X, const Matrix &T, int batch_size);
  BatchIterator begin() const;
  BatchIterator end() const;

private:
  const Matrix &X_;
  const Matrix &T_;
  int batch_size_;
};

class DataLoader {
public:
  DataLoader(const Matrix &X, const Matrix &T);
  void shuffle();
  BatchProxy batches(int batch_size) const;
  int numBatches(int batch_size) const;

private:
  Matrix X_;
  Matrix T_;
};

} // namespace NeuralNet
