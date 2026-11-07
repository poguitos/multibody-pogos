#pragma once

// JSON output for the comparison programs (plan task 5.3). Numbers are
// written with 17 significant digits, so every double reads back exactly.

#include <iomanip>
#include <sstream>
#include <string>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd_crosscheck {

inline std::string num(mbd::Real x)
{
    std::ostringstream o;
    o << std::setprecision(17) << x;
    return o.str();
}

template <class Vector>
std::string array(const Vector& x)
{
    std::string s = "[";
    for (Eigen::Index k = 0; k < x.size(); ++k) s += (k ? ", " : "") + num(x(k));
    return s + "]";
}

inline std::string matrix(const mbd::MatX& M)
{
    std::string s = "[";
    for (Eigen::Index r = 0; r < M.rows(); ++r) s += (r ? ", " : "") + array(mbd::VecX(M.row(r).transpose()));
    return s + "]";
}

/// A transform as {"q": quaternion (x, y, z, w), "p": origin}.
inline std::string transform(const mbd::Transform3& X)
{
    return "{\"q\": " + array(X.q.coeffs()) + ", \"p\": " + array(X.p) + "}";
}

/// Body b of the model: its parent, joint, coordinates, joint frames and
/// inertia, as compare_pinocchio.py and compare_chrono.py read them.
inline std::string body(const mbd::kernel::Model& model, int b)
{
    const auto i = static_cast<std::size_t>(b);
    const mbd::RigidBodyInertia& I = model.inertia[i];
    std::ostringstream o;
    o << "{\"body\": " << b << ", \"parent\": " << model.parent[i] << ", \"joint\": \"" << model.joint[i]->name()
      << "\", \"idx_q\": " << model.idx_q[i] << ", \"idx_v\": " << model.idx_v[i] << ", \"nq\": " << model.nqs[i]
      << ", \"nv\": " << model.nvs[i] << ", \"X_PJ\": " << transform(model.X_PJ[i])
      << ", \"X_CJ\": " << transform(model.X_CJ[i]) << ", \"mass\": " << num(I.mass)
      << ", \"com\": " << array(I.com_B) << ", \"I_com\": " << matrix(I.I_com_B) << "}";
    return o.str();
}

} // namespace mbd_crosscheck
