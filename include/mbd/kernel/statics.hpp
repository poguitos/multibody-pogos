#pragma once

// Static equilibrium (plan task 3.5).
//
// A system is in static equilibrium at q when, at rest, nothing accelerates:
// the applied forces, gravity and the constraint forces balance in every
// direction the constraints allow. With f(q) the generalized forces at rest
// (the force elements, joint forces, tau and force_callback of the
// simulator, less gravity's rnea(q, 0, 0)) and J the constraint Jacobian,
//
//     F(q, lambda) = f(q) + J^T lambda = 0,      phi(q, t) = 0,
//
// with time frozen at t: a driver holds its target, and its rates play no
// part. The residual is measured as the accelerations at rest, a = W^-1 F
// with J a = 0 (ConstraintSolver::accelerations_at_rest), as the plan's
// criterion asks: |a| below a tolerance in m/s^2 and rad/s^2.
//
// Newton's method works on the constraint surface. At q, with N an
// orthonormal basis of the motions the constraints allow (J N = 0, held
// coordinates excluded), the step is dq = N y with
//
//     (N^T K N) y = -N^T f,       K = dF/dq at fixed lambda,
//
// K by central differences of F (decision D24: the elements' analytic
// derivatives replace them later, term by term). N^T K N is the reduced
// tangent stiffness, with the constraint forces' geometric stiffness; its
// singular directions are those without any restoring force (a car on a flat
// road can roll forward, sideways and turn), which the step leaves alone.
// The step is halved until the accelerations decrease (in the norm a^T M a),
// and each trial point is projected back onto the constraints.
//
// When no step reduces them (a body hanging above the ground, where nothing
// pushes back yet), dynamic relaxation takes over: the system moves under
// the forces at rest, and its velocities are zeroed each time its kinetic
// energy passes a peak (kinetic damping), until the accelerations have
// fallen a hundredfold; Newton then resumes.
//
// At the end the reduced stiffness is examined: a direction of negative
// stiffness means the equilibrium is unstable (a pendulum balanced upright),
// and is reported.

#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

struct StaticsOptions {
    /// On the largest acceleration at rest, |a|_inf [m/s^2 or rad/s^2].
    Real tolerance{1e-6};
    /// Newton iterations.
    int max_iterations{50};
    /// Central-difference step for the tangent stiffness [m or rad].
    Real stiffness_step{1e-6};
    /// Velocity coordinates (0 to nv - 1) held at their values; the forces
    /// that would hold them are not asked for. A rotation is held whole.
    std::vector<int> hold;
    /// Fall back on dynamic relaxation when Newton cannot reduce the
    /// accelerations.
    bool dynamic_relaxation{true};
    /// Time step of the relaxation [s]; it must resolve the stiffest motion.
    Real relaxation_step{1e-3};
    /// Steps of relaxation allowed in all.
    int relaxation_max_steps{20000};
};

struct StaticsReport {
    bool converged{false};
    int iterations{0};              ///< Newton iterations
    int relaxations{0};             ///< Times Newton fell back on relaxation
    int relaxation_steps{0};        ///< Relaxation steps in all
    Real acceleration_before{0.0};  ///< |a|_inf at the given configuration (projected)
    Real acceleration{0.0};         ///< |a|_inf at the end
    int largest_acceleration_at{-1};  ///< Its velocity coordinate
    /// |a|_inf at the start of each Newton iteration, then at the end.
    std::vector<Real> history;
    int degrees_of_freedom{0};      ///< Motions the constraints and holds allow
    int neutral_directions{0};      ///< Of those, without stiffness at the end
    int unstable_directions{0};     ///< Of those, with negative stiffness at the end

    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<std::string> notes;

    bool ok() const { return errors.empty(); }

    /// The counts and the history, then every message, one per line.
    std::string summary(const Model& model) const;
};

/// Bring the simulator to static equilibrium at its time, from its q: q is
/// replaced by the equilibrium (or the best point reached), v is zeroed and
/// the kinematics refreshed. Throws MBD-K060 or MBD-K061 for a malformed
/// hold list; reports everything else.
StaticsReport static_equilibrium(Simulator& sim, const StaticsOptions& options = {});

} // namespace mbd::kernel
