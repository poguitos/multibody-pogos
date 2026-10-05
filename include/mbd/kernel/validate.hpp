#pragma once

// Checks of a system before it is simulated (plan task 2.9).
//
// validate() looks for the mistakes that otherwise surface later as a
// singular matrix, a NaN or a quietly wrong answer, and reports each one with
// a message that names the body, constraint or force element concerned. It
// also counts the degrees of freedom: the velocities of the tree less the
// independent constraint equations.

#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

struct ValidationReport {
    std::vector<std::string> errors;     ///< The system cannot be simulated as it is
    std::vector<std::string> warnings;   ///< It can, but probably not as meant
    std::vector<std::string> notes;      ///< Worth knowing, not a fault

    int bodies{0};                  ///< Not counting the ground
    int coordinates{0};             ///< nq
    int velocities{0};              ///< nv: the degrees of freedom of the tree
    int constraint_equations{0};
    int independent_equations{0};   ///< Rank of the constraint Jacobian
    int redundant_equations{0};
    int degrees_of_freedom{0};      ///< velocities - independent_equations

    bool ok() const { return errors.empty(); }

    /// The counts, then every message, one per line.
    std::string summary() const;
};

/// Check `sys` at its neutral configuration.
ValidationReport validate(const System& sys);

/// Check `sys` at the configuration q, at time t. The constraint counts and
/// the mass matrix are evaluated there.
ValidationReport validate(const System& sys, const VecX& q, Real t = 0.0);

} // namespace mbd::kernel
