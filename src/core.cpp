#include "core.hpp"
#include <spdlog/spdlog.h>

double compute_mean(const Eigen::VectorXd& v) {
    if (v.size() == 0) {
        spdlog::warn("compute_mean called with empty vector");
        return 0.0;
    }
    double sum = v.sum();
    double mean = sum / static_cast<double>(v.size());
    spdlog::info("compute_mean: size={} sum={} mean={}", v.size(), sum, mean);
    return mean;
}

std::string make_greeting(const std::string& name) {
    std::string msg = "Hello, " + name + " from multibody_core!";
    spdlog::info(msg);
    return msg;
}
