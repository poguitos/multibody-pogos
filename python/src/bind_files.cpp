// Model files (plan task 6.4, D56): a System as JSON and back.

#include <filesystem>
#include <string>

#include <pybind11/stl/filesystem.h>

#include "bindings.hpp"
#include "mbd/io/model_file.hpp"

namespace mbd::python {

void bind_files(py::module_& m)
{
    using namespace pybind11::literals;
    using kernel::System;

    m.def("model_to_json", &io::model_to_json, "system"_a,
          "The system as JSON text (docs/model-files.md, format version 1): every number reads back as the same "
          "double. What a file cannot hold (a UserForce, a time function given as code) raises MBD-I008.");
    m.def("model_from_json", &io::model_from_json, "text"_a,
          "The system written in `text`, by model_to_json() or by hand. An unknown key, a missing one or a "
          "wrong value raises an error naming its place in the text (MBD-I002 to I007).");
    m.def(
        "save_model",
        [](const System& system, const std::filesystem::path& path) { io::save_model(system, path.string()); },
        "system"_a, "path"_a, "model_to_json() written to the file at `path` (a str or a pathlib.Path).");
    m.def(
        "load_model", [](const std::filesystem::path& path) { return io::load_model(path.string()); }, "path"_a,
        "The system in the model file at `path` (a str or a pathlib.Path); see model_from_json().");
}

} // namespace mbd::python
