#pragma once

// Static equilibrium (plan task 3.5, decision D26).
//
// An equilibrium is a configuration q on the constraints at which the system,
// at rest, does not accelerate: a(q) = 0, with a the accelerations of the
// constrained dynamics at v = 0 and the time-dependent constraints frozen
// where they are at time t (their prescribed accelerations left out:
// statics has no inertia; ConstraintSolver::quasi_static). The forces are
// whatever the simulator applies at rest: gravity, force elements, joint
// forces, tau and the callbacks.
//
// Newton's method on a(q) = 0, in the tangent space of the constraints:
//   - G = da/dq by central differences, each perturbed configuration
//     projected back onto the constraints, so that G vanishes on the
//     directions across them;
//   - the step is the least, in the weights W (the diagonal of the mass
//     matrix), that zeroes G step + a. Through RankRevealingLDLT: with S =
//     W^1/2 and Gs = S G S^-1, step = S^-1 Gs^T y with (Gs Gs^T) y = -S a.
//     Directions with neither stiffness nor force, such as the position and
//     heading of a vehicle on tyres at rest, are not moved (to first order);
//     the report counts them;
//   - the step is shortened so that no translation changes by more than
//     max_translation and no rotation by more than max_rotation, then halved
//     until |S a| decreases.
// When no step decreases |S a| (a direction with force but no stiffness: a
// wheel above the ground), dynamic relaxation takes over: a motion driven by
// the accelerations at rest, with kinetic damping (the velocities set to zero
// each time the kinetic energy has passed a maximum), until those
// accelerations fall below relaxation_tolerance. Newton then resumes. The
// forces are those at rest: damping and tyre slip forces play no part in the
// equilibrium, and are too stiff at low speed for an explicit step.

#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

struct StaticOptions {
    /// On the largest acceleration at rest [m/s^2 or rad/s^2].
    Real tolerance{1e-6};
    /// Newton iterations, over all runs.
    int max_iterations{50};
    /// Step of the central differences [m or rad].
    Real perturbation{1e-6};
    /// The largest change of a translation and of a rotation coordinate in
    /// one Newton step [m, rad] (ADAMS's TLIMIT and ALIMIT): a longer step is
    /// shortened. Without them, a wheel hanging on a preloaded spring is sent
    /// in one step past the end of its suspension's travel.
    Real max_translation{0.02};
    Real max_rotation{10.0 * pi / 180.0};
    /// Time step of dynamic relaxation [s]; explicit, so below 2 / (the
    /// highest natural frequency in rad/s).
    Real relaxation_step{1e-3};
    /// Relaxation steps, over all runs.
    int max_relaxation_steps{20000};
    /// The largest acceleration at rest at which relaxation hands back to
    /// Newton.
    Real relaxation_tolerance{1e-2};
};

struct StaticReport {
    Real time{0.0};
    bool converged{false};            ///< Largest acceleration <= tolerance
    int iterations{0};                ///< Newton iterations
    int relaxation_runs{0};
    int relaxation_steps{0};
    Real residual_given{0.0};         ///< Largest acceleration at rest at the given positions, on the constraints
    Real residual{0.0};               ///< At the end
    std::vector<Real> history;        ///< Largest acceleration after each Newton iteration
    int degrees_of_freedom{0};
    int supported{0};                 ///< Directions with stiffness at the end: the rank of G
    int neutral{0};                   ///< degrees_of_freedom - supported
    VecX q_given, q;
    VecX position_change;             ///< difference(q_given, q), in velocity coordinates
    Loads loads;                      ///< At rest at the end, the drivers frozen: v_dot holds the accelerations

    /// What did not go as asked, and facts worth knowing, each with its code.
    std::vector<std::string> warnings;
    std::vector<std::string> notes;

    /// The counts, the history, each coordinate's change, the messages.
    std::string summary(const Model& model) const;
};

/// Find the static equilibrium nearest the simulator's positions at its time;
/// see the top of this file. The result is written to the simulator, at rest
/// (v = 0), even when not converged; the given positions stay in the report.
/// For setup: it allocates.
[[nodiscard]] StaticReport static_equilibrium(Simulator& sim, const StaticOptions& options = {});

} // namespace mbd::kernel
