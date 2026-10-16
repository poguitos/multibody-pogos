#pragma once

// The step-control checks and messages shared by the integrators with error
// control (simulator.cpp, simulator_implicit.cpp). Internal to the library.

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel::detail {

/// Throws MBD-K054 when the settings cannot be worked with.
void check_step_control(const StepControl& sc);

/// MBD-K055: the error control asks for a step below step_control.min_step.
[[noreturn]] void throw_step_too_short(const char* method, Real h, Real t, Real min_step, Real error);

/// MBD-K056: more than step_control.max_steps tries within one call to step().
[[noreturn]] void throw_too_many_steps(const char* method, int max_steps, Real dt, Real t);

/// MBD-K058: an implicit step that Newton could not solve.
[[noreturn]] void throw_unsolved(const char* method, Real h, Real t, bool regular, int max_iterations);

} // namespace mbd::kernel::detail
