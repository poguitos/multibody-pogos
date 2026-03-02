#pragma once

// Basic rigid-body dynamics and simple time integration for a single body.

#include <Eigen/Cholesky>
#include "mbd/rigid_body.hpp"

namespace mbd {

/// External forces and torques acting on a rigid body at its COM,
/// all expressed in the world frame W.
struct RigidBodyForces {
    Vec3 f_W{Vec3::Zero()};
    Vec3 tau_W{Vec3::Zero()};
};

/// Compute linear and angular acceleration for a single rigid body.
inline void compute_rigid_body_acceleration(
    const RigidBodyInertia& inertia,
    const RigidBodyState& state,
    const RigidBodyForces& forces,
    const Vec3& gravity_W,
    Vec3& a_W_out,
    Vec3& alpha_W_out)
{
    MBD_THROW_IF(inertia.mass <= Real(0.0),
        "compute_rigid_body_acceleration: mass must be > 0");

    // Linear acceleration: a = g + f_ext / m
    a_W_out = gravity_W + forces.f_W / inertia.mass;

    // Angular acceleration: Euler's equations in world frame
    // I_W * alpha = tau - w x (I_W * w)
    const Mat3 R_WB = state.q_WB.toRotationMatrix();
    const Mat3 I_W = R_WB * inertia.I_com_B * R_WB.transpose();
    const Vec3 L_W = I_W * state.w_WB;
    const Vec3 gyro_term = state.w_WB.cross(L_W);
    const Vec3 rhs = forces.tau_W - gyro_term;

    Eigen::LLT<Mat3> llt(I_W);
    MBD_THROW_IF(llt.info() != Eigen::Success,
        "compute_rigid_body_acceleration: inertia matrix not SPD");

    alpha_W_out = llt.solve(rhs);
}

/// Semi-implicit (symplectic) Euler integration for a single rigid body.
inline void integrate_rigid_body_semi_implicit(
    const RigidBodyInertia& inertia,
    const Vec3& gravity_W,
    const RigidBodyForces& forces,
    Real dt,
    RigidBodyState& state)
{
    MBD_ASSERT(dt > Real(0.0));

    Vec3 a_W, alpha_W;
    compute_rigid_body_acceleration(inertia, state, forces, gravity_W, a_W, alpha_W);

    // Update velocities first (semi-implicit)
    state.v_WB += a_W * dt;
    state.w_WB += alpha_W * dt;

    // Update positions using new velocities
    state.p_WB += state.v_WB * dt;

    if (state.w_WB.squaredNorm() > Real(0.0)) {
        state.q_WB = integrate_quat(state.q_WB, state.w_WB, dt);
    } else {
        state.q_WB = normalize_quat(state.q_WB);
    }
}

} // namespace mbd