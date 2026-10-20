#pragma once

// Linearisation (plan task 3.7, decision D27).
//
// About an operating point (q0, v0, t) of a simulator, usually a static
// equilibrium (v0 = 0), small motions obey, in reduced coordinates z on the
// constraints,
//
//   d/dt [z; z_dot] = A [z; z_dot],   A = [[0, I], [A_q, A_v]],
//
// with dq = N z and dv = N z_dot. The n columns of N (nv x n) span the motions
// the constraints allow: they are columns of the projector onto those motions
// that is orthogonal in the weights W (the diagonal of the mass matrix),
// picked independent by RankRevealingLDLT. A_q and A_v are the derivatives
// of the accelerations along them, by central differences, each perturbed
// position projected back onto the constraints, reduced by the W-weighted
// left inverse of N. Everything the simulator applies acts, as at the
// operating point. About a state in motion the same formulas are used; they
// are exact for joints whose coordinates add, while for spherical and free
// joints the moving frame adds terms that are not modelled.
//
// The eigenvalues of A are the modes: -zeta w +- i w sqrt(1 - zeta^2) for an
// oscillation of natural frequency w and damping ratio zeta, real ones for an
// overdamped motion, zero for a neutral one (a vehicle's position on the
// ground: a double zero, the position and its rate). The mode shapes are the
// position parts of the eigenvectors, mapped back by N to velocity
// coordinates. Each mode's kinetic energy is shared out among the velocity
// coordinates (task 5.1, D43): coordinate k holds
//
//   e_k = Re(conj(phi_k) (M phi)_k) / Re(phi^H M phi),
//
// M the mass matrix at the operating point and phi the shape; the shares
// add up to 1. It is the modal energy distribution used to judge how pure a
// mode is (the bounce, pitch and roll of a car on its springs, an engine on
// its mounts). Where M couples coordinates a share can be negative: the
// cross terms are split equally between the two coordinates.

#include <complex>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

struct LinearizationOptions {
    /// Step of the central differences [m or rad, and per second for the
    /// velocities].
    Real perturbation{1e-6};
};

/// One mode. A complex pair of eigenvalues is listed once, with its positive
/// imaginary part; a real eigenvalue on its own.
struct Mode {
    std::complex<Real> eigenvalue;
    Real natural_frequency{0.0};   ///< |eigenvalue| [rad/s]
    Real damping_ratio{0.0};       ///< -Re / |eigenvalue|; 0 for a zero eigenvalue
    Eigen::VectorXcd shape;        ///< dq in velocity coordinates, its largest entry 1
    /// The share of the mode's kinetic energy in each velocity coordinate,
    /// adding up to 1 (see the top of this file). Zero for a mode whose
    /// shape carries no kinetic energy.
    VecX energy_fraction;

    bool oscillatory() const { return eigenvalue.imag() > 0.0; }
};

struct LinearModel {
    Real time{0.0};
    VecX q, v;                     ///< The operating point, on the constraints
    Real residual{0.0};            ///< Largest acceleration there; zero at an equilibrium
    int degrees_of_freedom{0};     ///< n
    MatX basis;                    ///< N, nv x n
    MatX A;                        ///< 2n x 2n
    std::vector<Mode> modes;       ///< By natural frequency

    /// What makes the model doubtful, each with its code.
    std::vector<std::string> warnings;

    /// The operating point, each mode's frequency, damping and largest shape
    /// components, and the warnings.
    std::string summary(const Model& model) const;
};

/// Linearise the simulator's system about its state (q, v) at its time; see
/// the top of this file. The state is first moved onto the constraints if it
/// is not on them (with a warning). The simulator is left at that state. For
/// analysis: it allocates.
[[nodiscard]] LinearModel linearize(Simulator& sim, const LinearizationOptions& options = {});

} // namespace mbd::kernel
