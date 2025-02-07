#include "external/eigen-3.4.0/Eigen/Dense"
#include <functional>
#include <cmath>

struct LossFunction {
    std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> forward;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> backward;
};

inline LossFunction createMSE() {
    LossFunction mse;

    mse.forward = [](const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) -> double {
        Eigen::MatrixXd diff = predictions - targets;
        return diff.array().square().mean();
    };

    mse.backward = [](const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) -> Eigen::MatrixXd {
        double n = static_cast<double>(predictions.rows());
        return (2.0 / n) * (predictions - targets);
    };

    return mse;
}