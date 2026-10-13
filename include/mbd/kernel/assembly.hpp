#pragma once

// Initial-condition assembly (plan task 3.4, decision D25).
//
// The initial values of a model rarely satisfy its constraints: a loop is
// open by a few millimetres, a velocity would stretch a link. assemble()
// makes them consistent, keeping the coordinates the user holds exactly as
// given and changing the others as little as possible:
//
//   positions      min 1/2 d^T W d   with  phi(q_given (+) d, t) = 0,  d_H = 0
//   velocities     min 1/2 e^T W e   with  J (v_given + e) = nu,       e_H = 0
//   accelerations  the equations of motion at the assembled state, which give
//                  the accelerations and the constraint forces (compute_loads)
//
// d is the change of the coordinates in velocity coordinates, d =
// difference(q_given, q); e the change of the velocities; H the held
// velocity indices. W is the diagonal of the mass matrix, at the given
// positions for the positions and at the assembled positions for the
// velocities. Its entries carry each coordinate's own units (a mass for a
// slider, an inertia for a hinge), so the correction does not depend on the
// units chosen; and since W is diagonal, a coordinate that no constraint
// involves is never changed (the full mass matrix would turn a wheel's spin
// angle to keep the wheel still in space while its upright turns).
//
// Holding velocity index k keeps component k of d (or e) at zero. For a
// revolute, prismatic, cylindrical, universal or planar joint that keeps the
// coordinate at its given value. A spherical or free joint moves on its own
// frame (see JointModel::integrate): holding its three angular components
// keeps its orientation, holding a free joint's three linear components its
// position; hold_joint() holds all of them.
//
// The positions are found in two stages. The first is Gauss-Newton onto
// phi = 0, as in ConstraintSolver::project: each step the least change that
// zeroes the linearized constraints, halved until |phi| decreases. That
// reaches the constraints, but each step is the least from where it starts,
// so the sum of the steps is not the least correction. The second stage
// makes the whole correction the least: from q with d = difference(q_given,
// q), it takes the least d_new with J d_new = J d - phi and steps to it. Its
// fixed point has phi = 0 and W d = J^T mu, the conditions for the least
// correction. It converges linearly, the faster the smaller the correction
// is against the curvature of the constraints. It is exact for joints whose
// coordinates add (all but spherical and free); for those, d is measured by
// the logarithm of their relative rotation and the step treats it as
// additive, which is exact to second order in the rotation.

#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

/// What assemble() keeps and how far it iterates.
struct AssemblyOptions {
    /// Velocity indices whose positions are kept as given.
    std::vector<int> hold_position;
    /// Velocity indices whose velocities are kept as given.
    std::vector<int> hold_velocity;
    /// On |phi|, |J v - nu| and on the last step of the second position stage.
    Real tolerance{1e-10};
    /// For each of the two position stages.
    int max_iterations{50};

    /// Hold every coordinate of the joint above `body`, in position, velocity
    /// or both.
    AssemblyOptions& hold_joint(const Model& model, int body,
                                bool position = true, bool velocity = true);
};

/// What assemble() did, and the state it left.
struct AssemblyReport {
    Real time{0.0};
    int equations{0};             ///< Constraint equations
    int rank{0};                  ///< Independent equations, at the assembled positions
    int degrees_of_freedom{0};    ///< Velocities less the rank
    std::vector<int> held_positions;    ///< Sorted, as applied
    std::vector<int> held_velocities;   ///< Sorted, as applied
    int position_hold_rank{0};    ///< Rank of the constraints in the velocities not held in position
    int velocity_hold_rank{0};    ///< The same for the velocities not held in velocity

    // --- Positions -------------------------------------------------------------
    bool positions_assembled{false};   ///< |phi| <= tolerance
    bool smallest_correction{false};   ///< ... and the second stage converged
    int projection_iterations{0};      ///< First stage
    int refinement_iterations{0};      ///< Second stage
    Real position_residual_given{0.0}; ///< |phi| at the given positions
    Real position_residual{0.0};       ///< |phi| at the assembled positions
    VecX q_given, q;
    VecX position_change;              ///< difference(q_given, q), in velocity coordinates

    // --- Velocities ------------------------------------------------------------
    bool velocities_assembled{false};  ///< |J v - nu| <= tolerance
    Real velocity_residual_given{0.0};
    Real velocity_residual{0.0};
    VecX v_given, v;

    // --- Accelerations and forces, at the assembled state ----------------------
    Loads loads;
    Real acceleration_residual{0.0};   ///< |J v_dot - gamma|

    /// What did not go as asked, each with its code. Nothing is printed: the
    /// report is the result (as with validate()).
    std::vector<std::string> warnings;

    /// Positions and velocities satisfy the constraints.
    bool ok() const { return positions_assembled && velocities_assembled; }

    /// The counts, the residuals, every coordinate's given and assembled
    /// value, and the warnings, as text.
    std::string summary(const Model& model) const;
};

/// Assemble the simulator's state (q, v) at its time: see the top of this
/// file. The assembled state is written to the simulator, refreshed, even
/// when it is not fully assembled (the report then says why); the given one
/// stays in the report. For setup, not for the step: it allocates.
[[nodiscard]] AssemblyReport assemble(Simulator& sim, const AssemblyOptions& options = {});

} // namespace mbd::kernel
