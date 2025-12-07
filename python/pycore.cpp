#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pycore, m) {
    m.doc() = "Python bindings for multibody_core";

    m.def("compute_mean", &compute_mean,
          "Compute the mean of a 1D Eigen vector");

    m.def("make_greeting", &make_greeting,
          "Return a greeting string from C++ core");
}
