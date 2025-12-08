#pragma once

// Basic rigid-body dynamics and simple time integration for a single body.

#include <Eigen/Cholesky>      // for LLT solve
#include <mbd/rigid_body.hpp>  // includes core + math + inertia/state

namespace mbd {

// External forces and torques acting on a rigid body at its COM,
// all expressed in the world frame W.
struct RigidBodyForces
{
    Vec3 f_W{Vec3::Zero()};   // net force at COM [N], expressed in W
    Vec3 tau_W{Vec3::Zero()}; // net torque about COM [N·m], expressed in W
};

// Compute linear and angular acceleration for a single rigid body.
//
// a_W_out     : linear acceleration of body origin in W [m/s^2]
// alpha_W_out : angular acceleration of body frame in W [rad/s^2]
inline void compute_rigid_body_acceleration(
    const RigidBodyInertia& inertia,
    const RigidBodyState&   state,
    const RigidBodyForces&  forces,
    const Vec3&             gravity_W,
    Vec3&                   a_W_out,
    Vec3&                   alpha_W_out)
{
    MBD_THROW_IF(inertia.mass <= Real(0.0),
                 "compute_rigid_body_acceleration: mass must be > 0");

    // Linear acceleration: external forces + gravity as acceleration
    a_W_out = gravity_W + forces.f_W / inertia.mass;

    // Angular acceleration: tau = I_W * alpha  =>  alpha = I_W^{-1} * tau
    // Convert inertia from body frame B to world frame W:
    const RotMat3 R_WB = state.q_WB.toRotationMatrix();
    const Mat3    I_W  = R_WB * inertia.I_com_B * R_WB.transpose();

    Eigen::LLT<Mat3> llt(I_W);
    MBD_THROW_IF(llt.info() != Eigen::Success,
                 "compute_rigid_body_acceleration: inertia matrix not SPD");

    alpha_W_out = llt.solve(forces.tau_W);
}

// Semi-implicit (symplectic) Euler integration for a single rigid body.
//
// v_{n+1} = v_n + a_n dt
// w_{n+1} = w_n + alpha_n dt
// p_{n+1} = p_n + v_{n+1} dt
// q_{n+1} = q_n ⊗ exp( (w_{n+1} dt) )
inline void integrate_rigid_body_semi_implicit(
    const RigidBodyInertia& inertia,
    const Vec3&             gravity_W,
    const RigidBodyForces&  forces,
    Real                    dt,
    RigidBodyState&         state)
{
    MBD_ASSERT(dt > Real(0.0));

    Vec3 a_W, alpha_W;
    compute_rigid_body_acceleration(inertia, state, forces, gravity_W,
                                    a_W, alpha_W);

    // Update velocities
    state.v_WB += a_W * dt;
    state.w_WB += alpha_W * dt;

    // Update position using new velocity (semi-implicit Euler)
    state.p_WB += state.v_WB * dt;

    // Update orientation using new angular velocity
    if (state.w_WB.squaredNorm() > Real(0.0)) {
        state.q_WB = integrate_quat(state.q_WB, state.w_WB, dt);
    } else {
        // Keep quaternion normalized even if omega is zero
        state.q_WB = normalize_quat(state.q_WB);
    }
}

} // namespace mbd
