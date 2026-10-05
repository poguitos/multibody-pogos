#pragma once

// Kinematic analysis tools: position-level solver, geometric extraction,
// suspension sweep infrastructure.

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/simulator.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

namespace mbd {

// ============================================================================
// Geometric extraction from wheel body pose
// ============================================================================

/// Extract camber angle from a wheel body state.
///
/// Camber: inclination of the wheel spin axis (body Y) from vertical (world Y),
/// measured in the frontal plane (YZ).
///   Positive = top of wheel tilts toward +Z (outward for left-side wheel).
///   Zero = wheel perfectly vertical.
///
/// \param state  The wheel body's state (after FK).
Real extract_camber(const RigidBodyState& state);

/// Extract toe angle from a wheel body state.
///
/// Toe: yaw of the wheel forward direction (body X) relative to world X,
/// measured in the ground plane (XZ).
///   Positive = wheel points toward +Z.
///   For a left wheel: positive toe = toe-out.
///   For a right wheel: positive toe = toe-in.
///
/// \param state  The wheel body's state (after FK).
Real extract_toe(const RigidBodyState& state);

// ============================================================================
// Kinematic sweep result
// ============================================================================

/// One data point from a kinematic sweep.
struct KinematicSweepPoint {
    Real bump{0.0};     ///< Vertical travel [m], positive = compression (wheel up)
    Real camber{0.0};   ///< [rad]
    Real toe{0.0};      ///< [rad]
    Real wheel_y{0.0};  ///< Wheel center height [m]
    bool converged{false};
};

/// Result of a kinematic sweep.
struct KinematicSweepResult {
    std::vector<KinematicSweepPoint> points;

    /// Export to CSV file.
    void export_csv(const std::string& filename) const;

    /// Camber gain: average dcamber/dbump over the sweep [rad/m].
    Real camber_gain() const;
};

// ============================================================================
// Kinematic analysis on the kernel (plan task 2.7)
// ============================================================================
//
// The drivers of a mechanism (wheel height, steering rack) are constraints
// whose target depends on t. In a kinematic analysis t is therefore not a
// time but the driving parameter: a bump sweep solves phi(q, t) = 0 at
// successive values of t instead of editing a constraint's target.

/// The configuration of a kernel system, for kinematic analysis.
class Kinematics {
public:
    /// Starts at the neutral configuration, t = 0.
    explicit Kinematics(const kernel::System& sys);

    /// Solve phi(q, t) = 0, starting from the current q. True if converged.
    bool solve(int max_iterations = 100, Real tolerance = 1e-10);

    /// The constraint values phi(q, t).
    const VecX& phi();

    /// Velocities at (q, t) that keep the constraints satisfied, J v = nu.
    /// For a mechanism with no degrees of freedom left, these are the
    /// derivatives of the motion with respect to t, in velocity coordinates
    /// (a suspension's motion ratios, for instance); where freedom is left,
    /// the least-norm solution. Call after solve().
    const VecX& velocities();

    /// Accelerations at (q, v, t), J a = gamma, with v from velocities(): for
    /// a mechanism with no degrees of freedom left, the second derivatives of
    /// the motion with respect to t. Call after velocities().
    const VecX& accelerations();

    /// Recompute the body placements after changing q by hand.
    void update() { kernel::forward_kinematics(system_.model, data_, q, v_zero_); }

    /// World state of a body (at rest) at the current q.
    RigidBodyState state(int body) const { return kernel::body_state(data_, body); }

    const kernel::System& system() const { return system_; }

private:
    const kernel::System& system_;
    kernel::Data data_;
    kernel::ConstraintSolver solver_;
    VecX v_zero_;
    VecX v_, a_;

public:
    VecX q;          ///< Coordinates
    Real t{0.0};     ///< Driving parameter
};

/// Sweep a corner through vertical travel: the system's bump driver sets the
/// wheel-centre height to its nominal value plus t, so t is the bump travel.
/// Each point starts from the last converged one; k ends at the last point.
KinematicSweepResult sweep_bump_travel(Kinematics& k, int upright_body,
                                       Real bump_min, Real bump_max, int n_steps = 41);

/// A driver that holds the coordinate `axis` of a body point (in world axes)
/// at nominal + t: the bump prescription of a kinematic corner.
std::shared_ptr<const kernel::ConstraintModel> point_height_driver(
    int body, const Vec3& point_B, int axis, Real nominal);

} // namespace mbd
