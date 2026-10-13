#pragma once

// Debugging aids (documentation track H, task H.4): a system described as
// text, and a simulator's state dumped as text. The step trace is on the
// simulator (Simulator::trace_steps, trace(), write_trace()).

#include <string>

#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

/// The system as text: the counts (with validate()'s rank and degrees of
/// freedom at the neutral configuration, and its numbers of findings), the
/// gravity, every body with its joint, coordinates, velocities, mass and
/// centre of mass, every constraint, force element and joint force with the
/// bodies it acts on.
std::string describe(const System& sys);

/// The same, with validate()'s counts at the configuration q and time t.
std::string describe(const System& sys, const VecX& q, Real t = 0.0);

/// The simulator's state as text: time, coordinates and velocities, each
/// body's world placement and velocity, and the constraint residuals.
std::string dump_state(const Simulator& sim);

} // namespace mbd::kernel
