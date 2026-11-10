// The extension module multibody._mbd (plan Phase 6, D52). Imported by the
// package (python/multibody/__init__.py), which also defines the exception
// and warning classes that this file raises.

#include "bindings.hpp"

namespace mbd::python {
namespace {

/// The engine's warnings as Python warnings of class MbdWarning, so that
/// Python's filters apply to them (shown once, turned into errors, recorded
/// by pytest). The interpreter lock is taken in case the warning comes from
/// a thread that does not hold it.
void warning_to_python(const std::string& message)
{
    py::gil_scoped_acquire gil;
    const py::object category = py::module_::import("multibody._errors").attr("MbdWarning");
    py::module_::import("warnings").attr("warn")(message, category, 2);
}

} // namespace
} // namespace mbd::python

PYBIND11_MODULE(_mbd, m)
{
    using namespace mbd::python;
    m.doc() = "The C++ engine of the multibody package; import multibody, not this module.";
    m.attr("__version__") = MBD_VERSION;

    // mbd::MbdError -> multibody.MbdError, the message (and so its code) kept.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const mbd::MbdError& e) {
            const py::object error = py::module_::import("multibody._errors").attr("MbdError");
            PyErr_SetString(error.ptr(), e.what());
        }
    });
    // The sink is global: put the previous one back when the interpreter
    // exits, so that no warning is sent to a Python that is gone.
    static const mbd::DiagnosticSink previous_sink = mbd::diagnostic_sink();
    mbd::diagnostic_sink() = &warning_to_python;
    py::module_::import("atexit").attr("register")(
        py::cpp_function([] { mbd::diagnostic_sink() = previous_sink; }));

    bind_core(m);
    bind_model(m);
    bind_constraints(m);
    bind_forces(m);
    bind_simulator(m);
    bind_analysis(m);
    bind_files(m);
}
