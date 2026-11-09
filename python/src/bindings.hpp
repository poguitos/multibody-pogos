#pragma once

// The Python interface (plan Phase 6, D52): one extension module,
// multibody._mbd, bound in parts, one file per part. The classes keep their
// C++ names; vectors and matrices are NumPy arrays.

#include <string>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"

namespace py = pybind11;

namespace mbd::kernel {
class Simulator;
}

namespace mbd::python {

void bind_core(py::module_& m);
void bind_model(py::module_& m);
void bind_constraints(py::module_& m);
void bind_forces(py::module_& m);
void bind_simulator(py::module_& m);
void bind_analysis(py::module_& m);
/// Events and the force callback, on the Simulator class (bind_analysis.cpp).
void bind_simulator_callbacks(py::class_<kernel::Simulator>& sim);
/// State elements written in Python, attach() and the step trace (bind_state.cpp).
void bind_state_elements(py::module_& m, py::class_<kernel::Simulator>& sim);

/// A quaternion as NumPy sees it: [w, x, y, z].
using QuatArray = Eigen::Vector4d;
inline QuatArray to_array(const Quat& q) { return QuatArray(q.w(), q.x(), q.y(), q.z()); }
inline Quat to_quat(const QuatArray& a) { return Quat(a(0), a(1), a(2), a(3)); }

/// A property for an Eigen member of fixed size (a Vec3, a Mat3) that reads as
/// a writable NumPy view of the member, as the simulator's state does, so
/// that `model.gravity[2] = -1.62` writes the member. (pybind11's
/// def_readwrite gives a read-only view.) A value of another shape is
/// refused by the conversion.
template <class Cls, class C, class T>
void eigen_member(Cls& cls, const char* name, T C::*member, const char* doc = "")
{
    cls.def_property(
        name, py::cpp_function([member](C& c) -> T& { return c.*member; }, py::return_value_policy::reference_internal),
        [member](C& c, const T& value) { c.*member = value; }, doc);
}

/// target = value, if their sizes agree; MBD-P001 otherwise, naming `what`.
/// Assigning an Eigen vector of another size would resize the target, and a
/// simulator's state must keep the size of its model.
inline void assign_same_size(VecX& target, const VecX& value, const char* what)
{
    MBD_THROW_IF(value.size() != target.size(),
                 std::string("MBD-P001: ") + what + " has " + std::to_string(target.size())
                     + " entries; the value given has " + std::to_string(value.size()));
    target = value;
}

} // namespace mbd::python
