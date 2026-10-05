#pragma once

// Debugging aids (plan task H.4): what a model is, what state it is in, and
// what each step of a simulation did.
//
//   describe(system)      the bodies, joints, constraints, force elements and
//                         the count of degrees of freedom, as text;
//   dump_state(sim)       the simulator's state: coordinates, velocities and
//                         body placements, each constraint's residual;
//   Simulator::trace      one TraceRow per step (simulator.hpp): drift before
//                         projection, projection iterations and residuals,
//                         rank, energies, the power and work of the applied
//                         forces, and the energy balance;
//   write_trace_csv, read_trace_csv
//                         the trace as CSV, one row per step;
//   diagnose(trace)       what the trace says went wrong, with codes K100 to
//                         K104 (docs/help/messages.md).
//
// The energy balance is kinetic + gravity's potential - the work of the
// applied forces (force elements, joint forces, tau, callbacks), less its
// first value. Springs and dampers are in the work, so the balance needs no
// potential from any element: it stays at zero, to the accuracy of the
// integration and of the trapezoidal rule for the work, whatever the forces.
// When it does not, the integration or the projection is losing or making
// energy; when the energy rises and the balance does not move, the applied
// forces are putting it in.

#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

/// The system as text: one line per body (joint, parent, coordinate
/// indices, mass, centre of mass), constraint and force element, then the
/// counts of validate() at the neutral configuration.
std::string describe(const System& sys);

/// The same, with the counts at the configuration q and time t: a loop's
/// neutral configuration may be singular (a four-bar laid straight).
std::string describe(const System& sys, const VecX& q, Real t = 0.0);

/// The simulator's state as text: time; per body its joint's coordinates and
/// velocities, its world position, orientation (quaternion x, y, z, w) and
/// velocity; per constraint the largest |phi|; the last projection.
std::string dump_state(const Simulator& sim);

/// The trace as CSV, with a header of the TraceRow fields, 17 significant
/// digits so that it reads back exactly.
void write_trace_csv(const std::vector<TraceRow>& trace, const std::string& path);

/// A trace written by write_trace_csv. Throws MBD-A030 if the file cannot be
/// read or is not such a trace.
std::vector<TraceRow> read_trace_csv(const std::string& path);

/// What a trace says: warnings and notes with codes, each with the time it
/// first appeared.
struct TraceDiagnosis {
    std::vector<std::string> warnings;
    std::vector<std::string> notes;
    bool clean() const { return warnings.empty() && notes.empty(); }
    std::string summary() const;
};

/// Look through a trace for: projections that failed (K100), an energy
/// balance that drifts (K101), energy put in by the applied forces (K102),
/// redundant constraint equations (K103), constraints that drift far within
/// a step (K104).
TraceDiagnosis diagnose(const std::vector<TraceRow>& trace);

} // namespace mbd::kernel
