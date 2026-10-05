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
inline Real extract_camber(const RigidBodyState& state)
{
    const Vec3 spin_axis_W = state.q_WB * Vec3::UnitY();
    return std::atan2(spin_axis_W.z(), spin_axis_W.y());
}

/// Extract toe angle from a wheel body state.
///
/// Toe: yaw of the wheel forward direction (body X) relative to world X,
/// measured in the ground plane (XZ).
///   Positive = wheel points toward +Z.
///   For a left wheel: positive toe = toe-out.
///   For a right wheel: positive toe = toe-in.
///
/// \param state  The wheel body's state (after FK).
inline Real extract_toe(const RigidBodyState& state)
{
    const Vec3 fwd_W = state.q_WB * Vec3::UnitX();
    return std::atan2(fwd_W.z(), fwd_W.x());
}

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
    void export_csv(const std::string& filename) const
    {
        std::ofstream file(filename);
        file << std::fixed << std::setprecision(6);
        file << "bump_mm,camber_deg,toe_deg,wheel_center_y_mm,converged\n";

        for (const auto& p : points) {
            file << p.bump * 1000.0 << ","
                 << p.camber * 180.0 / 3.14159265358979323846 << ","
                 << p.toe * 180.0 / 3.14159265358979323846 << ","
                 << p.wheel_y * 1000.0 << ","
                 << (p.converged ? 1 : 0) << "\n";
        }
    }

    /// Camber gain: average dcamber/dbump over the sweep [rad/m].
    Real camber_gain() const
    {
        if (points.size() < 2) return 0.0;
        const auto& first = points.front();
        const auto& last  = points.back();
        Real dbump = last.bump - first.bump;
        if (std::abs(dbump) < 1e-12) return 0.0;
        return (last.camber - first.camber) / dbump;
    }
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
    explicit Kinematics(const kernel::System& sys)
        : system_(sys)
        , data_(sys.model)
        , solver_(sys.model, sys.constraints)
        , v_zero_(VecX::Zero(sys.model.nv))
        , q(sys.model.neutral_configuration())
    {
        update();
    }

    /// Solve phi(q, t) = 0, starting from the current q. True if converged.
    bool solve(int max_iterations = 100, Real tolerance = 1e-10)
    {
        VecX v = v_zero_;
        const kernel::ProjectionInfo info =
            solver_.project(data_, q, v, t, tolerance, max_iterations);
        update();
        return info.converged;
    }

    /// The constraint values phi(q, t).
    const VecX& phi()
    {
        solver_.evaluate(data_, q, v_zero_, t);
        return solver_.phi();
    }

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

public:
    VecX q;          ///< Coordinates
    Real t{0.0};     ///< Driving parameter
};

/// Sweep a corner through vertical travel: the system's bump driver sets the
/// wheel-centre height to its nominal value plus t, so t is the bump travel.
/// Each point starts from the last converged one; k ends at the last point.
inline KinematicSweepResult sweep_bump_travel(Kinematics& k, int upright_body,
                                              Real bump_min, Real bump_max, int n_steps = 41)
{
    KinematicSweepResult result;
    result.points.reserve(static_cast<std::size_t>(n_steps));
    VecX q_start = k.q;
    for (int i = 0; i < n_steps; ++i) {
        const Real bump = bump_min + (bump_max - bump_min) * i / std::max(n_steps - 1, 1);
        k.t = bump;
        k.q = q_start;
        const bool ok = k.solve();

        KinematicSweepPoint pt;
        pt.bump = bump;
        pt.converged = ok;
        const RigidBodyState s = k.state(upright_body);
        pt.wheel_y = s.p_WB.y();
        pt.camber = extract_camber(s);
        pt.toe = extract_toe(s);
        result.points.push_back(pt);

        if (ok) q_start = k.q;
    }
    return result;
}

/// A driver that holds the coordinate `axis` of a body point (in world axes)
/// at nominal + t: the bump prescription of a kinematic corner.
inline std::shared_ptr<const kernel::ConstraintModel> point_height_driver(
    int body, const Vec3& point_B, int axis, Real nominal)
{
    return std::make_shared<kernel::Dot2>(
        kernel::Marker{0, Transform3::Identity()}, axis,
        kernel::Marker{body, Transform3::FromTranslation(point_B)},
        kernel::TimeFunction([nominal](Real t) { return nominal + t; },
                             [](Real) { return 1.0; },
                             [](Real) { return 0.0; }));
}

} // namespace mbd