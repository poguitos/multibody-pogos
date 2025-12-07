#pragma once

#include <Eigen/Dense>
#include <string>

// Simple example function: compute mean of a vector.
double compute_mean(const Eigen::VectorXd& v);

// Example logging wrapper: returns a greeting string.
std::string make_greeting(const std::string& name);
