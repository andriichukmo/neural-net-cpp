#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <functional>
#include <vector>

namespace NeuralNet {

using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Signature = double(double);
using Function = std::function<Signature>;

} // namespace NeuralNet
