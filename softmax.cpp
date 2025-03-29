#include "softmax.h"
#include "types.h"

namespace NeuralNet {

Matrix softmax(const Matrix &x) {
  Matrix exp_x = x.array().exp();
  Eigen::RowVectorXd sum_exp = exp_x.colwise().sum();
  Matrix softmax_x = exp_x.array().rowwise() / sum_exp.array();
  return softmax_x;
}

} // namespace NeuralNet
