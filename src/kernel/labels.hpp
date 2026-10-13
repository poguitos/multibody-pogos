#pragma once

// How the kernel's messages and reports name things and write numbers.
// Internal to the library.

#include <cstdio>
#include <sstream>
#include <string>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel::detail {

/// "1.23e-04": residuals and tolerances in messages.
inline std::string sci(Real x)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.2e", x);
    return text;
}

/// "body 2 (coupler)", or "body 2" when it has no name or is not in the model.
inline std::string body_label(const Model& model, int i)
{
    std::string s = "body " + std::to_string(i);
    if (i >= 0 && i < static_cast<int>(model.name.size()) && !model.name[static_cast<std::size_t>(i)].empty()) {
        s += " (" + model.name[static_cast<std::size_t>(i)] + ")";
    }
    return s;
}

/// The body whose joint has velocity index k; 0 if none has.
inline int body_of_velocity(const Model& model, int k)
{
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto b = static_cast<std::size_t>(i);
        if (k >= model.idx_v[b] && k < model.idx_v[b] + model.nvs[b]) return i;
    }
    return 0;
}

/// "body 2 (coupler), revolute", with the component for a joint of several
/// velocities: "body 4 (upright), spherical 1".
inline std::string velocity_label(const Model& model, int k)
{
    const int body = body_of_velocity(model, k);
    const auto b = static_cast<std::size_t>(body);
    std::string s = body_label(model, body) + ", " + model.joint[b]->name();
    if (model.nvs[b] > 1) s += " " + std::to_string(k - model.idx_v[b]);
    return s;
}

/// The coordinate behind velocity k before and after: "1.3 -> 1.25 (change
/// -0.05)", or "1.3 (held)". A spherical or free joint, whose coordinates are
/// not its velocities' integrals, shows its change only: "change -0.05", or
/// "held". `change` is difference(q_given, q).
inline std::string position_text(const Model& model, int k, const VecX& q_given, const VecX& q,
                                 const VecX& change, bool held)
{
    const auto b = static_cast<std::size_t>(body_of_velocity(model, k));
    std::ostringstream os;
    os.precision(6);
    if (model.nqs[b] == model.nvs[b]) {
        const int iq = model.idx_q[b] + (k - model.idx_v[b]);
        os << q_given(iq);
        if (held) os << " (held)";
        else os << " -> " << q(iq) << " (change " << change(k) << ")";
    } else if (held) {
        os << "held";
    } else {
        os << "change " << change(k);
    }
    return os.str();
}

} // namespace mbd::kernel::detail
