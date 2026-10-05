#pragma once

// Linearisation about an operating point (plan task 3.7).
//
// A constrained system moves on its constraint surface, so it is linearised
// in coordinates of that surface. With N an orthonormal basis of the motions
// the constraints allow at the operating point (q0, v0) (J N = 0), the state
// is reduced to
//
//     y = N^T (q (-) q0),     z = N^T v,
//
// d of each for d degrees of freedom, and the state equations become
//
//     d/dt [y; z] = A [y; z] + B tau,
//
// tau the generalized forces added to the simulator's own (Simulator::tau).
// A is found by central differences: each reduced coordinate is moved by
// +-h, the configuration projected back onto the constraints and the
// velocities onto J v = nu, and the accelerations evaluated by the simulator
// with everything applied. Because the projection moves a point slightly,
// each column of A is fitted against the reduced coordinates the points
// actually have, not the ones asked for: A = dF dX^-1.
//
// At an operating point at rest in equilibrium (task 3.5), z = dy/dt and the
// second-order form follows:
//
//     M_r y'' + C_r y' + K_r y = B_r tau,     M_r = N^T M N,
//     K_r = -M_r dz'/dy,   C_r = -M_r dz'/dz.
//
// The modes are the eigenvalues of A: a pair sigma +- i omega_d is a
// vibration of natural frequency |lambda| / 2 pi and damping ratio
// -sigma / |lambda|. The undamped modes solve K_r phi = omega^2 M_r phi with
// the symmetric part of K_r. Mode shapes are given in the full velocity
// coordinates, N phi.

#include <complex>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

struct LinearizationOptions {
    /// Central-difference step on the reduced coordinates [m, rad, m/s,
    /// rad/s]; the error is of order step^2 and the roundoff eps / step.
    Real step{1e-5};
    /// Compute B, the response to each generalized force.
    bool inputs{true};
};

/// A mode of the linearised system, from an eigenvalue of A. Complex pairs
/// are listed once, with a positive imaginary part.
struct Mode {
    std::complex<Real> eigenvalue;
    Real natural_frequency_hz{0.0};   ///< |lambda| / 2 pi
    Real damped_frequency_hz{0.0};    ///< |Im lambda| / 2 pi
    Real damping_ratio{0.0};          ///< -Re lambda / |lambda|; 1 for a real negative eigenvalue
    /// Positions of the mode in the velocity coordinates (N times the
    /// eigenvector's y part), scaled so that its largest entry is 1.
    Eigen::Matrix<std::complex<Real>, Eigen::Dynamic, 1> shape;
    int largest_at{-1};               ///< Velocity coordinate of the largest entry
};

/// An undamped mode: K_r phi = omega^2 M_r phi.
struct UndampedMode {
    Real omega_squared{0.0};          ///< Negative for an unstable direction
    Real frequency_hz{0.0};           ///< sqrt(omega^2) / 2 pi; 0 if omega^2 <= 0
    VecX shape;                       ///< N phi, largest entry +1
    int largest_at{-1};
};

struct Linearization {
    MatX N;   ///< nv x d: the allowed motions; dq = N y, dv = N z
    MatX A;   ///< 2d x 2d
    MatX B;   ///< 2d x nv (empty if not asked for)
    MatX M;   ///< d x d: reduced mass N^T M N
    MatX K;   ///< d x d: reduced stiffness, -M dz'/dy
    MatX C;   ///< d x d: reduced damping, -M dz'/dz

    std::vector<Mode> modes;              ///< By natural frequency
    std::vector<UndampedMode> undamped;   ///< By omega^2

    Real operating_acceleration{0.0};     ///< |v_dot|_inf at the operating point
    Real operating_velocity{0.0};         ///< |v|_inf at the operating point

    std::vector<std::string> warnings;
    std::vector<std::string> notes;

    int degrees_of_freedom() const { return static_cast<int>(N.cols()); }

    /// The modes, one per line, then every message.
    std::string summary(const Model& model) const;
};

/// Linearise the simulator's system about its present state (q, v, time).
/// The state is left as it was. An operating point should be on the
/// constraints and, for modes, at rest in equilibrium (static_equilibrium);
/// otherwise the result says so (MBD-K080).
Linearization linearize(Simulator& sim, const LinearizationOptions& options = {});

} // namespace mbd::kernel
