#pragma once

// Tree-based dynamics algorithms: mass matrix, inverse dynamics (RNEA),
// and forward dynamics.
//
// All algorithms assume:
//   - Forward kinematics has been called (states[i] are up to date).
//   - Bodies are in topological order (parent index < child index).

#include "mbd/system.hpp"
#include "mbd/constraint.hpp"
#include <Eigen/Cholesky>

namespace mbd {

// ============================================================================
// Mass matrix via system Jacobians
// ============================================================================

/// Compute the n_dof x n_dof joint-space mass matrix M(q).
///
/// For each body, builds the 6 x n_dof Jacobian mapping q_dot to
/// [omega; v_com] in world frame, then accumulates M = sum J_i^T * I_sp * J_i.
///
/// Requires: forward kinematics already called.
inline MatX compute_mass_matrix(const MultibodySystem& sys)
{
    const int n = sys.total_dof;
    MatX M = MatX::Zero(n, n);

    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        const auto& inertia = sys.inertias[i];
        const auto& state   = sys.states[i];

        const Mat3 R_WB = state.q_WB.toRotationMatrix();
        const Vec3 c_W  = R_WB * inertia.com_B;
        const Vec3 p_com_W = state.p_WB + c_W;

        const Mat3 I_com_W = R_WB * inertia.I_com_B * R_WB.transpose();
        const Real m = inertia.mass;

        // Build 6 x n_dof Jacobian for body i (angular on top, linear on bottom)
        Eigen::Matrix<Real, 6, Eigen::Dynamic> J_i(6, n);
        J_i.setZero();

        // Walk from body i up through ancestors to ground
        BodyIndex b = i;
        while (b != kGroundIndex) {
            int j_idx = sys.body_infos[b].joint_idx;
            const auto& joint = *sys.joints[j_idx];
            int ndof = joint.num_dof();

            if (ndof > 0) {
                int offset = joint.q_offset;

                // Joint frame orientation in world
                const auto& ps = sys.states[joint.parent_body_idx];
                const Transform3 X_J = joint.joint_transform(sys.joint_q(j_idx));
                const Quat q_WJ = ps.q_WB * joint.X_PJ.q * X_J.q;
                const Mat3 R_WJ = q_WJ.toRotationMatrix();

                // Motion subspace in world
                const auto S_J = joint.motion_subspace(sys.joint_q(j_idx));
                const Eigen::Matrix<Real, 3, Eigen::Dynamic> S_ang_W = R_WJ * S_J.topRows(3);
                const Eigen::Matrix<Real, 3, Eigen::Dynamic> S_lin_W = R_WJ * S_J.bottomRows(3);

                // Joint position in world (parent-side attachment)
                const Vec3 p_joint_W = ps.pose_WB().apply(joint.X_PJ.p);
                const Vec3 r_joint_to_com = p_com_W - p_joint_W;

                // J_angular = S_ang_W
                // J_linear  = S_lin_W - skew(r_joint_to_com) * S_ang_W
                //   because: v_com += omega_joint x r = -r x omega = -skew(r) * omega
                J_i.block(0, offset, 3, ndof) = S_ang_W;
                J_i.block(3, offset, 3, ndof) = S_lin_W - skew(r_joint_to_com) * S_ang_W;
            }

            b = sys.body_infos[b].parent_idx;
        }

        // Accumulate M += J^T * diag(I_com_W, m*I3) * J
        const auto J_ang = J_i.topRows(3);
        const auto J_lin = J_i.bottomRows(3);

        M.noalias() += J_ang.transpose() * I_com_W * J_ang
                     +  m * (J_lin.transpose() * J_lin);
    }

    return M;
}

// ============================================================================
// Inverse dynamics (RNEA)
// ============================================================================

/// Compute the joint forces tau required to produce the given accelerations.
///
/// tau = M(q) * q_ddot + c(q, q_dot) - J^T * F_gravity
///
/// Uses the "gravity trick": ground acceleration is set to -gravity, so
/// gravity is automatically included in the equations.
///
/// Requires: forward kinematics already called.
inline VecX inverse_dynamics(const MultibodySystem& sys,
                             const VecX& q_ddot,
                             const Vec3& gravity)
{
    const int n_bodies = sys.body_count();
    const int n_dof    = sys.total_dof;

    // --- Per-body kinematic quantities ---
    std::vector<Vec3> w(n_bodies, Vec3::Zero());
    std::vector<Vec3> v(n_bodies, Vec3::Zero());
    std::vector<Vec3> al(n_bodies, Vec3::Zero());   // angular acceleration
    std::vector<Vec3> a(n_bodies, Vec3::Zero());     // linear acceleration of origin

    // Gravity trick: fictitious upward acceleration on ground
    a[kGroundIndex] = -gravity;

    // === Forward pass: propagate velocities and accelerations ===
    for (BodyIndex i = 1; i < n_bodies; ++i) {
        const auto& info  = sys.body_infos[i];
        const auto& joint = *sys.joints[info.joint_idx];
        const VecX q_j    = sys.joint_q(info.joint_idx);
        const VecX qd_j   = sys.joint_q_dot(info.joint_idx);

        const int ndof = joint.num_dof();
        const VecX qdd_j = (ndof > 0)
            ? q_ddot.segment(joint.q_offset, ndof)
            : VecX(0);

        // Joint frame orientation in world
        const auto& ps = sys.states[joint.parent_body_idx];
        const Transform3 X_J = joint.joint_transform(q_j);
        const Quat q_WJ = ps.q_WB * joint.X_PJ.q * X_J.q;
        const Mat3 R_WJ = q_WJ.toRotationMatrix();

        // Motion subspace in world
        const auto S_J = joint.motion_subspace(q_j);

        Vec3 omega_rel_W  = Vec3::Zero();
        Vec3 v_rel_W      = Vec3::Zero();
        Vec3 alpha_rel_W  = Vec3::Zero();
        Vec3 a_rel_lin_W  = Vec3::Zero();

        if (ndof > 0) {
            const auto S_ang = S_J.topRows(3);
            const auto S_lin = S_J.bottomRows(3);
            omega_rel_W  = R_WJ * (S_ang * qd_j);
            v_rel_W      = R_WJ * (S_lin * qd_j);
            alpha_rel_W  = R_WJ * (S_ang * qdd_j);
            a_rel_lin_W  = R_WJ * (S_lin * qdd_j);

            // Bias acceleration: accounts for dS/dt * q_dot when S depends on q
            const Vec6 c_bias = joint.bias_acceleration(q_j, qd_j);
            alpha_rel_W += R_WJ * c_bias.head<3>();
            a_rel_lin_W += R_WJ * c_bias.tail<3>();
        }

        // Angular velocity and acceleration
        const Vec3& w_p  = w[info.parent_idx];
        const Vec3& al_p = al[info.parent_idx];

        w[i]  = w_p + omega_rel_W;
        al[i] = al_p + alpha_rel_W + w_p.cross(omega_rel_W);

        // Linear velocity and acceleration — two-step propagation
        // Step 1: joint point (as point rigidly on parent body)
        const Vec3 p_joint_W = ps.pose_WB().apply(joint.X_PJ.p);
        const Vec3 r_P_to_J  = p_joint_W - sys.states[info.parent_idx].p_WB;

        const Vec3 v_J = v[info.parent_idx] + w_p.cross(r_P_to_J);
        const Vec3 a_J = a[info.parent_idx]
                       + al_p.cross(r_P_to_J)
                       + w_p.cross(w_p.cross(r_P_to_J));

        // Step 2: from moving joint frame to child body origin
        const Vec3 p_Jmoving_W = sys.states[i].pose_WB().apply(joint.X_CJ.p);
        const Vec3 r_J_to_Oi   = sys.states[i].p_WB - p_Jmoving_W;

        const Vec3 v_Jmoving = v_J + v_rel_W;
        v[i] = v_Jmoving + w[i].cross(r_J_to_Oi);

        a[i] = a_J
             + a_rel_lin_W
             + w[i].cross(v_rel_W)
             + al[i].cross(r_J_to_Oi)
             + w[i].cross(w[i].cross(r_J_to_Oi));
    }

    // === Backward pass: Newton-Euler forces and generalized forces ===
    // f_net[i] and tau_net[i] accumulate the total wrench (about body origin)
    // needed to produce the motion of body i and all its descendants.
    std::vector<Vec3> f_net(n_bodies, Vec3::Zero());
    std::vector<Vec3> tau_net(n_bodies, Vec3::Zero());

    VecX tau = VecX::Zero(n_dof);

    for (BodyIndex i = n_bodies - 1; i >= 1; --i) {
        const auto& inertia = sys.inertias[i];
        const auto& state   = sys.states[i];

        const Mat3 R_WB   = state.q_WB.toRotationMatrix();
        const Vec3 c_W    = R_WB * inertia.com_B;
        const Mat3 I_com_W = R_WB * inertia.I_com_B * R_WB.transpose();
        const Real m       = inertia.mass;

        // COM acceleration (from body origin acceleration)
        const Vec3 a_com = a[i] + al[i].cross(c_W) + w[i].cross(w[i].cross(c_W));

        // Newton-Euler at COM
        const Vec3 f_body   = m * a_com;
        const Vec3 tau_body = I_com_W * al[i] + w[i].cross(I_com_W * w[i]);

        // Wrench about body origin = Newton-Euler at COM transported to origin
        // tau_origin = tau_com + c_W x f
        f_net[i]   += f_body;
        tau_net[i] += tau_body + c_W.cross(f_body);

        // --- Project onto joint axis to get generalized force ---
        const auto& info  = sys.body_infos[i];
        const auto& joint = *sys.joints[info.joint_idx];
        const int ndof    = joint.num_dof();

        if (ndof > 0) {
            // Joint position in world
            const auto& parent_state = sys.states[joint.parent_body_idx];
            const Vec3 p_joint_W = parent_state.pose_WB().apply(joint.X_PJ.p);

            // Transport wrench from body origin to joint point
            // tau_at_joint = tau_origin + (p_origin - p_joint) x f
            const Vec3 r_origin_minus_joint = state.p_WB - p_joint_W;
            const Vec3 tau_at_joint = tau_net[i] + r_origin_minus_joint.cross(f_net[i]);

            // Motion subspace in world
            const VecX q_j = sys.joint_q(info.joint_idx);
            const Transform3 X_J = joint.joint_transform(q_j);
            const Quat q_WJ = parent_state.q_WB * joint.X_PJ.q * X_J.q;
            const Mat3 R_WJ = q_WJ.toRotationMatrix();

            const auto S_J     = joint.motion_subspace(q_j);
            const auto S_ang_W = (R_WJ * S_J.topRows(3)).eval();
            const auto S_lin_W = (R_WJ * S_J.bottomRows(3)).eval();

            // Generalized force: tau_gen = S_ang^T * tau_at_joint + S_lin^T * f
            tau.segment(joint.q_offset, ndof) =
                S_ang_W.transpose() * tau_at_joint
              + S_lin_W.transpose() * f_net[i];
        }

        // --- Propagate wrench to parent (about parent's origin) ---
        const BodyIndex p_idx = info.parent_idx;
        const Vec3 r_parent_to_i = state.p_WB - sys.states[p_idx].p_WB;

        f_net[p_idx]   += f_net[i];
        tau_net[p_idx] += tau_net[i] + r_parent_to_i.cross(f_net[i]);
    }

    return tau;
}

// ============================================================================
// Forward dynamics
// ============================================================================

/// Compute joint accelerations from applied generalized forces.
///
///   q_ddot = M(q)^{-1} * (tau_applied - h(q, q_dot))
///
/// where h = RNEA(q, q_dot, 0, gravity) is the bias (Coriolis + gravity).
///
/// Requires: forward kinematics already called.
inline VecX forward_dynamics(const MultibodySystem& sys,
                             const VecX& tau_applied,
                             const Vec3& gravity)
{
    const MatX M = compute_mass_matrix(sys);
    const VecX h = inverse_dynamics(sys, VecX::Zero(sys.total_dof), gravity);

    // Solve M * q_ddot = tau_applied - h
    Eigen::LLT<MatX> llt(M);
    MBD_THROW_IF(llt.info() != Eigen::Success,
        "forward_dynamics: mass matrix not SPD (degenerate configuration?)");

    return llt.solve(tau_applied - h);
}

// ============================================================================
// Body Jacobian at body origin (for constraint coupling)
// ============================================================================

/// Jacobian mapping q_dot to body-origin linear velocity and angular velocity.
/// Both expressed in world frame.
struct BodyJacobian {
    Eigen::Matrix<Real, 3, Eigen::Dynamic> J_omega;  // 3 x n_dof
    Eigen::Matrix<Real, 3, Eigen::Dynamic> J_v;      // 3 x n_dof (at body origin)
};

/// Compute the body Jacobian at the body origin for a given body.
/// Ground returns zero Jacobians.
///
/// Requires: FK already called.
inline BodyJacobian compute_body_jacobian_origin(
    const MultibodySystem& sys, BodyIndex body_idx)
{
    const int n = sys.total_dof;
    BodyJacobian bj;
    bj.J_omega = Eigen::Matrix<Real, 3, Eigen::Dynamic>::Zero(3, n);
    bj.J_v     = Eigen::Matrix<Real, 3, Eigen::Dynamic>::Zero(3, n);

    if (sys.is_ground(body_idx)) return bj;

    const Vec3& p_origin_W = sys.states[body_idx].p_WB;

    BodyIndex b = body_idx;
    while (b != kGroundIndex) {
        int j_idx = sys.body_infos[b].joint_idx;
        const auto& joint = *sys.joints[j_idx];
        int ndof = joint.num_dof();

        if (ndof > 0) {
            int offset = joint.q_offset;

            const auto& ps = sys.states[joint.parent_body_idx];
            const Transform3 X_J = joint.joint_transform(sys.joint_q(j_idx));
            const Quat q_WJ = ps.q_WB * joint.X_PJ.q * X_J.q;
            const Mat3 R_WJ = q_WJ.toRotationMatrix();

            const auto S_J = joint.motion_subspace(sys.joint_q(j_idx));
            const auto S_ang_W = (R_WJ * S_J.topRows(3)).eval();
            const auto S_lin_W = (R_WJ * S_J.bottomRows(3)).eval();

            const Vec3 p_joint_W = ps.pose_WB().apply(joint.X_PJ.p);
            const Vec3 r = p_origin_W - p_joint_W;

            bj.J_omega.block(0, offset, 3, ndof) = S_ang_W;
            bj.J_v.block(0, offset, 3, ndof)     = S_lin_W - skew(r) * S_ang_W;
        }

        b = sys.body_infos[b].parent_idx;
    }

    return bj;
}

// ============================================================================
// Body accelerations (kinematic forward pass, NO gravity trick)
// ============================================================================

/// Per-body acceleration at the body origin.
struct BodyAcceleration {
    Vec3 a;      // linear acceleration of body origin in world
    Vec3 alpha;  // angular acceleration in world
};

/// Compute actual body accelerations given q, q_dot, q_ddot.
///
/// This runs the RNEA forward pass without the gravity trick
/// (ground acceleration = 0), so the returned accelerations are
/// the real physical accelerations in the world frame.
///
/// Requires: FK already called.
inline std::vector<BodyAcceleration> compute_body_accelerations(
    const MultibodySystem& sys,
    const VecX& q_ddot)
{
    const int n_bodies = sys.body_count();

    std::vector<Vec3> w(n_bodies, Vec3::Zero());
    std::vector<Vec3> v_o(n_bodies, Vec3::Zero());
    std::vector<Vec3> al(n_bodies, Vec3::Zero());
    std::vector<Vec3> a_o(n_bodies, Vec3::Zero());

    // NO gravity trick: ground has zero acceleration.
    a_o[kGroundIndex] = Vec3::Zero();

    for (BodyIndex i = 1; i < n_bodies; ++i) {
        const auto& info  = sys.body_infos[i];
        const auto& joint = *sys.joints[info.joint_idx];
        const VecX q_j    = sys.joint_q(info.joint_idx);
        const VecX qd_j   = sys.joint_q_dot(info.joint_idx);

        const int ndof = joint.num_dof();
        const VecX qdd_j = (ndof > 0)
            ? q_ddot.segment(joint.q_offset, ndof)
            : VecX(0);

        const auto& ps = sys.states[joint.parent_body_idx];
        const Transform3 X_J = joint.joint_transform(q_j);
        const Quat q_WJ = ps.q_WB * joint.X_PJ.q * X_J.q;
        const Mat3 R_WJ = q_WJ.toRotationMatrix();

        const auto S_J = joint.motion_subspace(q_j);

        Vec3 omega_rel_W  = Vec3::Zero();
        Vec3 v_rel_W      = Vec3::Zero();
        Vec3 alpha_rel_W  = Vec3::Zero();
        Vec3 a_rel_lin_W  = Vec3::Zero();

        if (ndof > 0) {
            const auto S_ang = S_J.topRows(3);
            const auto S_lin = S_J.bottomRows(3);
            omega_rel_W  = R_WJ * (S_ang * qd_j);
            v_rel_W      = R_WJ * (S_lin * qd_j);
            alpha_rel_W  = R_WJ * (S_ang * qdd_j);
            a_rel_lin_W  = R_WJ * (S_lin * qdd_j);

            const Vec6 c_bias = joint.bias_acceleration(q_j, qd_j);
            alpha_rel_W += R_WJ * c_bias.head<3>();
            a_rel_lin_W += R_WJ * c_bias.tail<3>();
        }

        const Vec3& w_p  = w[info.parent_idx];
        const Vec3& al_p = al[info.parent_idx];

        w[i]  = w_p + omega_rel_W;
        al[i] = al_p + alpha_rel_W + w_p.cross(omega_rel_W);

        // Two-step linear propagation (matches FK velocity propagation)
        const Vec3 p_joint_W = ps.pose_WB().apply(joint.X_PJ.p);
        const Vec3 r_P_to_J  = p_joint_W - sys.states[info.parent_idx].p_WB;

        const Vec3 v_J = v_o[info.parent_idx] + w_p.cross(r_P_to_J);
        const Vec3 a_J = a_o[info.parent_idx]
                       + al_p.cross(r_P_to_J)
                       + w_p.cross(w_p.cross(r_P_to_J));

        const Vec3 p_Jmoving_W = sys.states[i].pose_WB().apply(joint.X_CJ.p);
        const Vec3 r_J_to_Oi   = sys.states[i].p_WB - p_Jmoving_W;

        const Vec3 v_Jmoving = v_J + v_rel_W;
        v_o[i] = v_Jmoving + w[i].cross(r_J_to_Oi);

        a_o[i] = a_J
               + a_rel_lin_W
               + w[i].cross(v_rel_W)
               + al[i].cross(r_J_to_Oi)
               + w[i].cross(w[i].cross(r_J_to_Oi));
    }

    std::vector<BodyAcceleration> result(n_bodies);
    for (int i = 0; i < n_bodies; ++i) {
        result[i].a     = a_o[i];
        result[i].alpha = al[i];
    }
    return result;
}

// ============================================================================
// Constrained forward dynamics (tree + loop-closing constraints)
// ============================================================================

/// Compute joint accelerations for a system with both tree joints and
/// loop-closing constraints (hybrid formulation).
///
/// When no constraints are present, falls through to unconstrained
/// forward_dynamics.
///
/// Baumgarte stabilization parameters:
///   alpha: velocity-level damping (critical damping ~ 5-20)
///   beta:  position-level stiffness (same range)
///
/// Requires: FK already called.
inline VecX constrained_forward_dynamics(
    const MultibodySystem& sys,
    const VecX& tau_applied,
    const Vec3& gravity,
    Real alpha = 5.0,
    Real beta  = 5.0)
{
    // Fall through to unconstrained dynamics when no constraints
    if (sys.constraints.empty()) {
        return forward_dynamics(sys, tau_applied, gravity);
    }

    const int n_dof = sys.total_dof;

    // --- Step 1: Unconstrained forward dynamics ---
    const MatX M = compute_mass_matrix(sys);
    const VecX h = inverse_dynamics(sys, VecX::Zero(n_dof), gravity);

    Eigen::LLT<MatX> M_llt(M);
    MBD_THROW_IF(M_llt.info() != Eigen::Success,
        "constrained_forward_dynamics: mass matrix not SPD");

    const VecX q_ddot_free = M_llt.solve(tau_applied - h);

    // --- Step 2: Count total constraint equations ---
    int total_eqs = 0;
    for (const auto& c : sys.constraints) {
        total_eqs += c->equation_count();
    }
    if (total_eqs == 0) return q_ddot_free;

    // --- Step 3: Build joint-space constraint Jacobian J_q ---
    //
    // Constraint Jacobians are n_eq x 6 with layout [v_x v_y v_z | w_x w_y w_z].
    // Body Jacobians map q_dot to [v_origin] and [omega] separately.
    // J_q = J_abs_lin * J_body_v + J_abs_ang * J_body_omega  (for each body pair)
    MatX J_q = MatX::Zero(total_eqs, n_dof);

    int row = 0;
    for (const auto& c : sys.constraints) {
        int n_eq = c->equation_count();

        Eigen::MatrixXd J1_abs, J2_abs;
        c->jacobian(sys, J1_abs, J2_abs);

        BodyJacobian bj1 = compute_body_jacobian_origin(sys, c->body1_idx);
        BodyJacobian bj2 = compute_body_jacobian_origin(sys, c->body2_idx);

        J_q.block(row, 0, n_eq, n_dof) =
            J1_abs.leftCols(3)  * bj1.J_v     +
            J1_abs.rightCols(3) * bj1.J_omega +
            J2_abs.leftCols(3)  * bj2.J_v     +
            J2_abs.rightCols(3) * bj2.J_omega;

        row += n_eq;
    }

    // --- Step 4: Compute Phi_ddot at q_ddot_free (actual body accelerations) ---
    //
    // Body accelerations from kinematic forward pass (no gravity trick).
    // Gravity is already baked into q_ddot_free via the RNEA bias h.
    const auto body_acc = compute_body_accelerations(sys, q_ddot_free);

    VecX phi_ddot_free = VecX::Zero(total_eqs);
    VecX phi_total     = VecX::Zero(total_eqs);

    row = 0;
    for (const auto& c : sys.constraints) {
        int n_eq = c->equation_count();

        Eigen::MatrixXd J1_abs, J2_abs;
        c->jacobian(sys, J1_abs, J2_abs);

        Eigen::VectorXd gamma;
        c->velocity_bias(sys, gamma);

        // Body accelerations as 6-vectors: [a_lin; alpha]
        Vec6 a1_6;
        a1_6 << body_acc[c->body1_idx].a, body_acc[c->body1_idx].alpha;
        Vec6 a2_6;
        a2_6 << body_acc[c->body2_idx].a, body_acc[c->body2_idx].alpha;

        phi_ddot_free.segment(row, n_eq) = J1_abs * a1_6 + J2_abs * a2_6 + gamma;

        // Constraint position error
        Eigen::VectorXd phi;
        c->evaluate(sys, phi);
        phi_total.segment(row, n_eq) = phi;

        row += n_eq;
    }

    // --- Step 5: Baumgarte stabilization ---
    const VecX phi_dot = J_q * sys.q_dot;
    const VecX stab = 2.0 * alpha * phi_dot + beta * beta * phi_total;

    // --- Step 6: Solve for Lagrange multipliers ---
    //
    // J_q * M^-1 * J_q^T * lambda = -(phi_ddot_free + stab)
    const MatX M_inv_JqT = M_llt.solve(J_q.transpose());
    const MatX A = J_q * M_inv_JqT;
    const VecX rhs = -(phi_ddot_free + stab);

    const VecX lambda = A.ldlt().solve(rhs);

    // --- Step 7: Corrected accelerations ---
    return q_ddot_free + M_inv_JqT * lambda;
}
// ============================================================================
// Force element projection: body-frame forces → joint-space generalized forces
// ============================================================================

/// Project the accumulated body-frame forces (from ForceElement::apply())
/// into joint-space generalized forces using the body-origin Jacobians.
///
/// tau_joint += J_v_i^T * f_W_i + J_omega_i^T * tau_W_i  for each body i.
///
/// This implements the principle of virtual work: the generalized force
/// corresponding to a body wrench is the wrench projected onto the
/// joint-space velocities.
///
/// Requires: FK already called, forces[] populated by ForceElement::apply().
inline VecX project_body_forces_to_joint_space(const MultibodySystem& sys)
{
    VecX tau = VecX::Zero(sys.total_dof);

    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        const auto& f = sys.forces[i];

        // Skip bodies with negligible forces (avoids unnecessary Jacobian computation)
        if (f.f_W.squaredNorm() < Real(1e-30) &&
            f.tau_W.squaredNorm() < Real(1e-30)) {
            continue;
        }

        const BodyJacobian bj = compute_body_jacobian_origin(sys, i);
        tau.noalias() += bj.J_v.transpose()     * f.f_W
                       + bj.J_omega.transpose()  * f.tau_W;
    }

    return tau;
}
} // namespace mbd