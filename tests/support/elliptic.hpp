#pragma once

// Elliptic integrals and Jacobi's elliptic functions, for the closed-form
// solutions of pendulums and of the torque-free body (plan tasks 2.7, 5.1,
// 5.2). Written here from the arithmetic-geometric mean, so that the tests'
// references do not depend on a library.

#include <cmath>

#include "mbd/core/core.hpp"

namespace mbd_test {

/// Complete elliptic integral of the first kind, K(k), by the
/// arithmetic-geometric mean: K = pi / (2 AGM(1, sqrt(1 - k^2))).
inline mbd::Real elliptic_K(mbd::Real k)
{
    mbd::Real a = 1.0, b = std::sqrt(1.0 - k * k);
    for (int i = 0; i < 30 && std::abs(a - b) > 1e-16; ++i) {
        const mbd::Real m = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = m;
    }
    return mbd::pi / (2.0 * a);
}

/// Jacobi's amplitude am(u | m) for the parameter m = k^2 (0 <= m < 1), by
/// the descending Landen transformation (Abramowitz and Stegun 16.4), and
/// dn(u | m) beside it. sn = sin(am), cn = cos(am). Valid for any real u: am
/// grows by pi over each 2 K(k).
inline mbd::Real jacobi_amplitude(mbd::Real u, mbd::Real m, mbd::Real& dn)
{
    mbd::Real a[32], c[32];
    a[0] = 1.0;
    mbd::Real b = std::sqrt(1.0 - m);
    c[0] = std::sqrt(m);
    int n = 0;
    while (std::abs(c[n]) > 1e-16 && n < 30) {
        a[n + 1] = 0.5 * (a[n] + b);
        c[n + 1] = 0.5 * (a[n] - b);
        b = std::sqrt(a[n] * b);
        ++n;
    }
    mbd::Real phi = std::ldexp(a[n] * u, n);   // 2^n a_n u
    mbd::Real phi_next = phi;
    for (int i = n; i > 0; --i) {
        phi_next = phi;
        phi = 0.5 * (phi + std::asin(c[i] / a[i] * std::sin(phi)));
    }
    dn = n > 0 ? std::cos(phi) / std::cos(phi_next - phi) : 1.0;
    return phi;
}

/// Jacobi's elliptic functions sn, cn and dn of u for the parameter m = k^2.
inline void jacobi(mbd::Real u, mbd::Real m, mbd::Real& sn, mbd::Real& cn, mbd::Real& dn)
{
    const mbd::Real phi = jacobi_amplitude(u, m, dn);
    sn = std::sin(phi);
    cn = std::cos(phi);
}

} // namespace mbd_test
