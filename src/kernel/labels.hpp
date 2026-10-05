#pragma once

// How the kernel's reports name bodies and constraints and print numbers:
// one place, so that validate() and assemble() read alike.

#include <cstddef>
#include <sstream>
#include <string>

#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel::labels {

/// "body 3 (coupler)", or "body 3" when it has no name.
inline std::string body(const Model& model, int i)
{
    std::string s = "body " + std::to_string(i);
    if (i >= 0 && i < static_cast<int>(model.name.size()) && !model.name[i].empty()) {
        s += " (" + model.name[i] + ")";
    }
    return s;
}

/// "constraint 2 (revolute)".
inline std::string constraint(std::size_t k, const ConstraintModel& c)
{
    return "constraint " + std::to_string(k) + " (" + c.name() + ")";
}

/// Four significant digits.
inline std::string number(Real x)
{
    std::ostringstream os;
    os.precision(4);
    os << x;
    return os.str();
}

} // namespace mbd::kernel::labels
