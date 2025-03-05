#pragma once

#include "external/eigen-3.4.0/Eigen/Dense"

namespace NeuralNet {

using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Signature = double(double);
using Function = std::function<Signature>;

} // namespace NeuralNet
