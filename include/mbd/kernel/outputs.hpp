#pragma once

// Output requests (plan task 3.3): the loads a simulation is asked for. Joint
// reactions, the constraint forces on each body, the accelerations; the
// energies are in algorithms.hpp (kinetic, gravitational) and on the force
// elements (potential_energy).

#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd {
class Recorder;   // analysis/recorder.hpp
}

namespace mbd::kernel {

/// The spatial force [moment; force] each joint transmits from the parent to
/// its child body, in the child-side joint frame and about its origin:
/// reactions[i] for the joint of body i (reactions[0] is unused). It includes
/// the joint's own actuation (joint forces, drivers, tau), whose component
/// along the joint's motion is the generalized force at the joint.
///
/// Needs data after forward_kinematics(q, v, v_dot), the accelerations being
/// those of the motion, and the wrenches external to the tree on each body:
/// applied forces and constraint forces, in world axes, the moment about the
/// world origin. Recursive Newton-Euler with external forces.
void joint_reactions(const Model& model, Data& data, const std::vector<Vec6>& external_W,
                     std::vector<Vec6>& reactions);

/// Loads at a simulator's state.
struct Loads {
    VecX v_dot;                         ///< Accelerations
    VecX lambda;                        ///< Constraint multipliers, as solved
    std::vector<Vec6> constraint_W;     ///< Constraint wrench on each body: world axes, moment about the world origin
    std::vector<Vec6> joint_reaction;   ///< See joint_reactions()
};

/// Evaluates the accelerations at the simulator's state (q, v, time), then the
/// constraint forces on each body and the joint reactions. For output, not
/// for the step: it allocates.
Loads compute_loads(Simulator& sim);

/// The drift channels (task 4.7): adds to `recorder` the channels
/// "constraint_drift" and "constraint_velocity_drift" (what each step left
/// before the projection) and "constraint_residual" and
/// "constraint_velocity_residual" (the state's), read from
/// sim.constraint_drift() at every sample. The simulator must outlive the
/// recorder's sampling.
void add_drift_channels(Recorder& recorder, const Simulator& sim);

} // namespace mbd::kernel
