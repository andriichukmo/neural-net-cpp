#pragma once

#include "external/eigen/Eigen/Core"
#include "external/eigen/Eigen/Dense"
#include <cassert>
#include <functional>

namespace NeuralNet {

using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Signature = double(double);
using Function = std::function<Signature>;

} // namespace NeuralNet
