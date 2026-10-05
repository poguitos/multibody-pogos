#pragma once

// Assembly of initial conditions (plan task 3.4).
//
// The values a model starts from rarely satisfy its constraints exactly: a
// loop is closed by hand at approximate angles, a suspension is set at its
// design angles, one coordinate is given a velocity and the others are left
// at zero. assemble() moves them onto the constraints, keeping the
// coordinates chosen as held exactly as given and changing the others as
// little as possible, and reports what it did:
//
//   positions      phi(q, t) = 0            least |q (-) q_given|
//   velocities     J v = nu                 least |v - v_given|
//   accelerations  J a = gamma(q, v, t)     least |a - a_given|
//
// "Least" is in the kinetic-energy metric: the mass matrix, at the given
// configuration for positions and at the assembled one for the rates, over
// the coordinates that are not held. Heavy bodies are therefore moved less
// than light ones, and the result does not depend on units. For positions
// the least correction is reached exactly for the scalar coordinates
// (revolute, prismatic, universal, cylindrical, planar joints); for the
// rotations of spherical and free joints the correction is measured as a
// rotation vector, and it is the least to second order in its size.
//
// Holding is by velocity coordinate (0 to nv - 1): the velocities of a joint
// are its coordinates' rates, and a position correction is a velocity
// applied for unit time. A rotation is held whole: either all three angular
// coordinates of a spherical or free joint, or none. The translation of a
// free joint may be held in part only when its rotation is held; its
// components are then along the joint's child-side axes.

#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

/// What assemble() keeps as given, and how closely it solves.
struct AssemblySpec {
    /// Velocity coordinates (0 to nv - 1) kept exactly as given, per level.
    std::vector<int> hold_positions;
    std::vector<int> hold_velocities;
    std::vector<int> hold_accelerations;

    /// On |phi| (lengths, or the dimensionless dot products of the
    /// orientation equations), |J v - nu| and |J a - gamma|.
    Real tolerance{1e-10};
    /// Gauss-Newton steps, and separately refinements towards the least
    /// correction.
    int max_iterations{50};

    /// Hold every coordinate of the joint of `body`, at every level.
    AssemblySpec& hold(const Model& model, int body);
    /// Hold coordinate k (0 to the joint's nv - 1) of the joint of `body`, at
    /// every level.
    AssemblySpec& hold(const Model& model, int body, int k);
};

/// How one level of the assembly went.
struct AssemblyLevel {
    bool done{false};              ///< This level was assembled
    int held{0};                   ///< Coordinates held
    Real residual_before{0.0};     ///< |phi|, |J v - nu| or |J a - gamma| as given
    Real residual{0.0};            ///< The same, assembled
    Real largest_change{0.0};      ///< Largest change of one coordinate (as a velocity coordinate)
    int largest_change_at{-1};     ///< Its index, or -1 if nothing changed
    int iterations{0};             ///< Positions: Gauss-Newton steps in all
    int refinements{0};            ///< Positions: steps towards the least correction
    int independent_equations{0};  ///< Rank of the constraints in the coordinates not held
    int freedom_left{0};           ///< Degrees of freedom the held coordinates do not fix
    bool converged{false};
};

/// What assemble() did, with every problem it met.
struct AssemblyReport {
    int velocities{0};              ///< nv
    int constraint_equations{0};
    int independent_equations{0};   ///< Rank of the constraint Jacobian, assembled
    int degrees_of_freedom{0};      ///< velocities - independent_equations

    AssemblyLevel position, velocity, acceleration;

    std::vector<std::string> errors;     ///< A level that could not be assembled
    std::vector<std::string> warnings;   ///< Held coordinates that fight the constraints
    std::vector<std::string> notes;      ///< Freedom left to the least correction

    bool ok() const { return errors.empty(); }

    /// The counts, one line per level, then every message, one per line.
    /// `model` names the coordinates that changed most.
    std::string summary(const Model& model) const;
};

/// Assemble q and v at time t. Throws (MBD-K060, MBD-K061) if a held index is
/// out of range or holds part of a rotation; reports everything else.
AssemblyReport assemble(const System& sys, VecX& q, VecX& v, Real t,
                        const AssemblySpec& spec = {});

/// Assemble q, v and the accelerations a at time t.
AssemblyReport assemble(const System& sys, VecX& q, VecX& v, VecX& a, Real t,
                        const AssemblySpec& spec = {});

/// Assemble the simulator's q and v at its time, then refresh its kinematics.
/// A failure is also reported as a warning (MBD-K067), since a simulation
/// started from it would begin with a jump.
AssemblyReport assemble(Simulator& sim, const AssemblySpec& spec = {});

/// The joint coordinate behind velocity index k, for messages: "coordinate
/// <j> of the <joint> joint of <body>".
std::string coordinate_label(const Model& model, int k);

} // namespace mbd::kernel
