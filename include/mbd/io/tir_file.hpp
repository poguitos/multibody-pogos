#pragma once

// Tyre property files (.tir) for the Magic Formula tyre (plan task 7.4,
// mbd/forces/magic_formula.hpp).
//
// A .tir file is text in sections, [SECTION], of KEY = value lines; '$'
// begins a comment, as does a line beginning with '!'; strings are quoted.
// Lines without '=' (the rows of a table, such as [SHAPE]'s) are skipped.
//
//   - [MODEL] FITTYP chooses the formulas: 6 or 21, MF 5.2; 61 or 62, MF 6.1
//     (MBD-I022 otherwise). A file without FITTYP may name them as Adams
//     does, PROPERTY_FILE_FORMAT = 'PAC2002' or 'MF_05' (MF 5.2), 'MF_61' or
//     'MF_62' (MF 6.1).
//   - [UNITS] must be SI: meter, newton, radians, kg, second, pascal
//     (MBD-I021 otherwise). Signs are taken as the file gives them: ISO.
//   - FITTYP (or PROPERTY_FILE_FORMAT), FNOMIN and UNLOADED_RADIUS are
//     required (MBD-I023); every other
//     parameter takes its default when left out (magic_formula.hpp).
//   - Keys are matched by name, whatever their section. A key the tyre does
//     not use (a SWIFT parameter, TYRESIDE, the file's header) is listed in
//     TirFile::ignored, not refused: .tir files carry many.

#include <string>
#include <vector>

#include "mbd/forces/magic_formula.hpp"

namespace mbd::io {

/// What a .tir file gave.
struct TirFile {
    MagicFormulaParameters parameters;
    std::vector<std::string> ignored;   ///< Keys present but not used, as SECTION.KEY
};

/// The parameters in .tir text; `name` names it in messages.
[[nodiscard]] TirFile parse_tir(const std::string& text, const std::string& name = "tir text");

/// parse_tir() of the file at `path` (MBD-I020 if it cannot be read).
[[nodiscard]] TirFile read_tir(const std::string& path);

} // namespace mbd::io
