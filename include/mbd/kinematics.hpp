#pragma once

// Kinematic analysis tools: position-level solver, geometric extraction,
// suspension sweep infrastructure.

#include "mbd/system.hpp"
#include "mbd/constraint.hpp"
#include "mbd/algorithms.hpp"

#include <Eigen/QR>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

namespace mbd {

// ============================================================================
// Constraint evaluation and Jacobian (reusable utilities)
// ============================================================================

/// Evaluate the total constraint violation vector Phi(q).
inline VecX evaluate_all_constraints(const MultibodySystem& sys)
{
    int total = 0;
    for (const auto& c : sys.constraints) {
        total += c->equation_count();
    }

    VecX phi(total);
    int row = 0;
    for (const auto& c : sys.constraints) {
        int n = c->equation_count();
        Eigen::VectorXd phi_c;
        c->evaluate(sys, phi_c);
        phi.segment(row, n) = phi_c;
        row += n;
    }
    return phi;
}

/// Compute the joint-space constraint Jacobian J_q (n_constraints x n_dof).
inline MatX compute_constraint_jacobian_q(const MultibodySystem& sys)
{
    int total = 0;
    for (const auto& c : sys.constraints) {
        total += c->equation_count();
    }

    MatX J_q = MatX::Zero(total, sys.total_dof);

    int row = 0;
    for (const auto& c : sys.constraints) {
        int n = c->equation_count();

        Eigen::MatrixXd J1_abs, J2_abs;
        c->jacobian(sys, J1_abs, J2_abs);

        BodyJacobian bj1 = compute_body_jacobian_origin(sys, c->body1_idx);
        BodyJacobian bj2 = compute_body_jacobian_origin(sys, c->body2_idx);

        J_q.block(row, 0, n, sys.total_dof) =
            J1_abs.leftCols(3)  * bj1.J_v     +
            J1_abs.rightCols(3) * bj1.J_omega  +
            J2_abs.leftCols(3)  * bj2.J_v     +
            J2_abs.rightCols(3) * bj2.J_omega;

        row += n;
    }
    return J_q;
}

// ============================================================================
// Position-level constraint solver (Newton-Raphson)
// ============================================================================

/// Solve for joint coordinates q such that all constraints Phi(q) = 0.
///
/// Uses Newton-Raphson with QR decomposition and backtracking line search
/// for robustness with nonlinear geometries (e.g., multi-link suspension).
///
/// Returns true if converged within tolerance.
inline bool solve_position_kinematics(MultibodySystem& sys,
                                      int max_iter = 100,
                                      Real tol = 1e-10)
{
    for (int iter = 0; iter < max_iter; ++iter) {
        sys.compute_forward_kinematics();

        VecX phi = evaluate_all_constraints(sys);
        Real err = phi.norm();
        if (err < tol) return true;

        MatX J_q = compute_constraint_jacobian_q(sys);

        Eigen::ColPivHouseholderQR<MatX> qr(J_q);
        VecX dq = qr.solve(-phi);

        // Backtracking line search: halve step until residual decreases
        VecX q_save = sys.q;
        Real step = 1.0;
        for (int ls = 0; ls < 12; ++ls) {
            sys.q = q_save + step * dq;
            sys.compute_forward_kinematics();
            VecX phi_new = evaluate_all_constraints(sys);
            if (phi_new.norm() < err) {
                break;
            }
            step *= 0.5;
        }
    }

    // Final check
    sys.compute_forward_kinematics();
    VecX phi = evaluate_all_constraints(sys);
    return phi.norm() < tol;
}

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

/// Sweep a suspension corner through vertical travel and record kinematics.
///
/// \param sys            The MultibodySystem (must have the suspension built).
/// \param bump_constraint Index into sys.constraints of the PointCoordinateConstraint
///                        that prescribes the wheel center Y position.
/// \param upright_body   BodyIndex of the upright/wheel whose pose is measured.
/// \param nominal_y      Wheel center Y at zero bump [m].
/// \param bump_min       Most negative bump (droop) [m], e.g. -0.05.
/// \param bump_max       Most positive bump (compression) [m], e.g. +0.05.
/// \param n_steps        Number of sweep points.
///
/// Assumes the system is at a valid configuration (q near the reference).
inline KinematicSweepResult sweep_bump_travel(
    MultibodySystem& sys,
    size_t bump_constraint_idx,
    BodyIndex upright_body,
    Real nominal_y,
    Real bump_min,
    Real bump_max,
    int n_steps = 41)
{
    KinematicSweepResult result;
    result.points.reserve(n_steps);

    VecX q_save = sys.q;

    auto* height_con = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[bump_constraint_idx].get());
    MBD_THROW_IF(!height_con, "sweep_bump_travel: constraint is not PointCoordinateConstraint");

    for (int i = 0; i < n_steps; ++i) {
        Real bump = bump_min + (bump_max - bump_min) * i / std::max(n_steps - 1, 1);

        // Set target height: nominal_y - bump (bump>0 = wheel moves up = y decreases)
        height_con->target = nominal_y + bump;
        // Start from saved configuration for robustness
        sys.q = q_save;

        bool ok = solve_position_kinematics(sys);

        KinematicSweepPoint pt;
        pt.bump = bump;
        pt.converged = ok;
        pt.wheel_y = sys.states[upright_body].p_WB.y();
        pt.camber = extract_camber(sys.states[upright_body]);
        pt.toe = extract_toe(sys.states[upright_body]);

        result.points.push_back(pt);

        // Use this solution as starting point for the next step
        if (ok) {
            q_save = sys.q;
        }
    }

    return result;
}

} // namespace mbd