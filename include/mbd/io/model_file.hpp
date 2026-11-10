#pragma once

// Model files (plan task 6.4, decision D56): a kernel::System, its tree, its
// constraints, its force elements and its joint forces, as JSON, and back.
// The format, version 1, is described in docs/model-files.md.
//
//   - Every number is written so that it reads back as the same double, and
//     a system read from a file writes the same text again: a model
//     survives a round trip unchanged, and so does every result computed
//     from it.
//   - Only what a file can hold is written: a time function given as code,
//     a force written as code (UserForce) and the vehicle layer's elements
//     are refused, by name (MBD-I008), rather than left out.
//   - Reading checks every key: an unknown one (a misspelling) is an error,
//     never ignored, and every error names the place in the file
//     ("bodies[2].X_PJ.quaternion"). The values then go through the same
//     constructors as a model built in code, with their own checks.

#include <string>

#include "mbd/kernel/simulator.hpp"

namespace mbd::io {

/// The system as JSON text (format "multibody model", version 1).
[[nodiscard]] std::string model_to_json(const kernel::System& system);

/// The system written in `text` by model_to_json(), or by hand.
[[nodiscard]] kernel::System model_from_json(const std::string& text);

/// model_to_json() written to the file at `path` (MBD-I001 if it cannot be).
void save_model(const kernel::System& system, const std::string& path);

/// model_from_json() of the file at `path` (MBD-I001 if it cannot be read).
[[nodiscard]] kernel::System load_model(const std::string& path);

} // namespace mbd::io
