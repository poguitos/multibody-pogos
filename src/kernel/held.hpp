#pragma once

// The lists of held coordinates of assembly and statics (tasks 3.4, 3.5):
// checked once, in one place.

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"

#include "labels.hpp"

namespace mbd::kernel::held {

/// The list sorted and without repeats. Throws MBD-K060 if an index is not a
/// velocity coordinate, MBD-K061 if it holds part of a rotation: a spherical
/// or free joint's three angular coordinates are held all or none, and a free
/// joint's translation in part only when its rotation is held.
inline std::vector<int> checked(const Model& model, const std::vector<int>& given,
                                const char* function, const char* what)
{
    std::vector<int> held = given;
    std::sort(held.begin(), held.end());
    held.erase(std::unique(held.begin(), held.end()), held.end());
    for (int k : held) {
        MBD_THROW_IF(k < 0 || k >= model.nv,
                     std::string("MBD-K060: ") + function + ": the held " + what + " include "
                         + std::to_string(k) + ", but the velocity coordinates are 0 to "
                         + std::to_string(model.nv - 1) + ".");
    }
    auto count = [&](int first, int n) {
        return static_cast<int>(std::count_if(held.begin(), held.end(),
                                              [&](int k) { return k >= first && k < first + n; }));
    };
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto& joint = model.joint[static_cast<std::size_t>(i)];
        if (!joint) continue;
        const char* name = joint->name();
        const bool is_spherical = std::strcmp(name, "spherical") == 0;
        const bool is_free = std::strcmp(name, "free") == 0;
        if (!is_spherical && !is_free) continue;
        const int first = model.idx_v[static_cast<std::size_t>(i)];
        const int angular = count(first, 3);
        const int linear = is_free ? count(first + 3, 3) : 0;
        MBD_THROW_IF((angular != 0 && angular != 3) || (angular == 0 && linear != 0 && linear != 3),
                     std::string("MBD-K061: ") + function + ": the held " + what + " hold part of the "
                         + name + " joint of " + labels::body(model, i) + " ("
                         + std::to_string(angular) + " of its 3 angular coordinates, "
                         + std::to_string(linear)
                         + " of its linear ones). Hold its rotation whole, or not at all; its "
                           "translation may be held in part only when the rotation is held.");
    }
    return held;
}

} // namespace mbd::kernel::held
