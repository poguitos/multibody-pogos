#pragma once

// Shared machinery for the invariant tests.
//
// An invariant is a relation that must hold for every joint, constraint and
// state, independent of any particular model:
//
//   - body velocities are the time derivative of body poses;
//   - the mass matrix, the inverse dynamics and Lagrange's equations describe
//     the same system;
//   - a constraint's Jacobian and acceleration term are the first and second
//     time derivatives of its position equation;
//   - energy and momentum are conserved when nothing dissipates them.
//
// Each check below compares the engine against finite differences of its own
// position-level quantities, which are the simplest and best-tested layer.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "mbd/model/system.hpp"
#include "mbd/model/constraint.hpp"
#include "mbd/algorithms/dynamics.hpp"
#include "mbd/integrators/simulator.hpp"

#include "support/rng.hpp"

namespace mbd_test {

using mbd::BodyIndex;
using mbd::MatX;
using mbd::MultibodySystem;
using mbd::Quat;
using mbd::Real;
using mbd::Transform3;
using mbd::Vec3;
using mbd::Vec6;
using mbd::VecX;

// ============================================================================
// Model construction
// ============================================================================

enum class JointKind { Revolute, Prismatic, Spherical, Universal, Free, Fixed };

inline const char* joint_name(JointKind k)
{
    switch (k) {
        case JointKind::Revolute:  return "revolute";
        case JointKind::Prismatic: return "prismatic";
        case JointKind::Spherical: return "spherical";
        case JointKind::Universal: return "universal";
        case JointKind::Free:      return "free";
        case JointKind::Fixed:     return "fixed";
    }
    return "?";
}

inline std::unique_ptr<mbd::Joint> make_joint(JointKind k,
                                              const Transform3& X_PJ,
                                              const Transform3& X_CJ,
                                              BodyIndex parent,
                                              BodyIndex child)
{
    switch (k) {
        case JointKind::Revolute:
            return std::make_unique<mbd::RevoluteCoordJoint>(X_PJ, X_CJ, parent, child);
        case JointKind::Prismatic:
            return std::make_unique<mbd::PrismaticCoordJoint>(X_PJ, X_CJ, parent, child);
        case JointKind::Spherical:
            return std::make_unique<mbd::SphericalCoordJoint>(X_PJ, X_CJ, parent, child);
        case JointKind::Universal:
            return std::make_unique<mbd::UniversalCoordJoint>(X_PJ, X_CJ, parent, child);
        case JointKind::Free:
            return std::make_unique<mbd::FreeCoordJoint>(X_PJ, X_CJ, parent, child);
        case JointKind::Fixed:
            return std::make_unique<mbd::FixedJoint>(X_PJ, X_CJ, parent, child);
    }
    return nullptr;
}

/// Add a body with a generic inertia (including a centre-of-mass offset),
/// attached to `parent` by a joint of the given kind with generic joint frames.
inline BodyIndex add_random_body(MultibodySystem& sys, Rng& rng, JointKind kind,
                                 BodyIndex parent, const std::string& name)
{
    const Real mass = rng.range(0.5, 3.0);
    const Vec3 half(rng.range(0.05, 0.3), rng.range(0.05, 0.3), rng.range(0.05, 0.3));
    mbd::RigidBodyInertia inertia = mbd::RigidBodyInertia::from_solid_box(mass, half);
    inertia.com_B = rng.vec(0.1);

    const BodyIndex b = sys.add_body(inertia, mbd::RigidBodyState{}, name, parent);
    const Transform3 X_PJ = rng.frame(0.4);
    const Transform3 X_CJ = rng.frame(0.3);
    sys.add_joint(make_joint(kind, X_PJ, X_CJ, parent, b));
    return b;
}

/// ground -> (root) -> (child) -> revolute. The third body shows whether an
/// error made at the second joint propagates down the chain.
inline void build_chain(MultibodySystem& sys, Rng& rng, JointKind root, JointKind child)
{
    const BodyIndex a = add_random_body(sys, rng, root, mbd::kGroundIndex, "a");
    const BodyIndex b = add_random_body(sys, rng, child, a, "b");
    add_random_body(sys, rng, JointKind::Revolute, b, "c");
}

/// Generic coordinates and velocities, well away from any singularity.
inline void randomize_state(MultibodySystem& sys, Rng& rng)
{
    for (int i = 0; i < sys.total_dof; ++i) {
        sys.q(i)     = rng.range(-0.6, 0.6);
        sys.q_dot(i) = rng.range(-1.2, 1.2);
    }
    sys.compute_kinematics();
}

// ============================================================================
// Mechanical quantities computed from first principles
// ============================================================================

inline Real kinetic_energy(const MultibodySystem& sys)
{
    const MatX M = mbd::compute_mass_matrix(sys);
    return Real(0.5) * sys.q_dot.dot(M * sys.q_dot);
}

inline Vec3 com_position(const MultibodySystem& sys, BodyIndex i)
{
    return sys.states[i].p_WB + sys.states[i].q_WB * sys.inertias[i].com_B;
}

inline Real potential_energy(const MultibodySystem& sys, const Vec3& gravity)
{
    Real V = 0.0;
    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        V -= sys.inertias[i].mass * gravity.dot(com_position(sys, i));
    }
    return V;
}

inline Vec3 linear_momentum(const MultibodySystem& sys)
{
    Vec3 P = Vec3::Zero();
    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        const auto& s = sys.states[i];
        const Vec3 c_W = s.q_WB * sys.inertias[i].com_B;
        P += sys.inertias[i].mass * (s.v_WB + s.w_WB.cross(c_W));
    }
    return P;
}

/// Angular momentum about the world origin.
inline Vec3 angular_momentum(const MultibodySystem& sys)
{
    Vec3 H = Vec3::Zero();
    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        const auto& s = sys.states[i];
        const mbd::Mat3 R = s.q_WB.toRotationMatrix();
        const Vec3 c_W = R * sys.inertias[i].com_B;
        const Vec3 v_com = s.v_WB + s.w_WB.cross(c_W);
        const mbd::Mat3 I_W = R * sys.inertias[i].I_com_B * R.transpose();
        H += I_W * s.w_WB + sys.inertias[i].mass * (s.p_WB + c_W).cross(v_com);
    }
    return H;
}

// ============================================================================
// Check 1: velocities against finite differences of the poses
// ============================================================================

struct KinematicsErrors {
    Real v_fk{0.0};   ///< velocity pass, linear
    Real w_fk{0.0};   ///< velocity pass, angular
    Real v_jac{0.0};  ///< body Jacobian, linear
    Real w_jac{0.0};  ///< body Jacobian, angular
    Real scale{1.0};  ///< 1 + largest speed seen, for relative tolerances
};

inline KinematicsErrors kinematics_errors(MultibodySystem& sys)
{
    const VecX q0 = sys.q, qd = sys.q_dot;
    const Real h = 1e-6;
    const int nb = sys.body_count();
    std::vector<Vec3> pp(nb), pm(nb);
    std::vector<Quat> rp(nb), rm(nb);

    sys.q = q0 + h * qd;
    sys.compute_forward_kinematics();
    for (int i = 0; i < nb; ++i) { pp[i] = sys.states[i].p_WB; rp[i] = sys.states[i].q_WB; }
    sys.q = q0 - h * qd;
    sys.compute_forward_kinematics();
    for (int i = 0; i < nb; ++i) { pm[i] = sys.states[i].p_WB; rm[i] = sys.states[i].q_WB; }

    sys.q = q0;
    sys.q_dot = qd;
    sys.compute_kinematics();

    KinematicsErrors e;
    for (int i = 1; i < nb; ++i) {
        const Vec3 v_num = (pp[i] - pm[i]) / (2 * h);
        Quat dq = rp[i] * rm[i].conjugate();
        if (dq.w() < 0) dq.coeffs() *= -1.0;
        const Vec3 w_num = 2.0 * dq.vec() / (2 * h);

        const mbd::BodyJacobian bj = mbd::compute_body_jacobian_origin(sys, i);
        e.v_fk  = std::max(e.v_fk,  (sys.states[i].v_WB - v_num).norm());
        e.w_fk  = std::max(e.w_fk,  (sys.states[i].w_WB - w_num).norm());
        e.v_jac = std::max(e.v_jac, (bj.J_v * qd - v_num).norm());
        e.w_jac = std::max(e.w_jac, (bj.J_omega * qd - w_num).norm());
        e.scale = std::max(e.scale, Real(1.0) + std::max(v_num.norm(), w_num.norm()));
    }
    return e;
}

// ============================================================================
// Check 2: mass matrix, inverse dynamics and Lagrange's equations agree
// ============================================================================

/// Velocity-dependent generalized force derived from the mass matrix alone:
///   h_k = sum_j (dM/dt)_kj q_dot_j - d/dq_k (1/2 q_dot^T M q_dot)
/// (Lagrange's equations with no potential).
inline VecX lagrangian_bias(MultibodySystem& sys)
{
    const VecX q0 = sys.q, qd = sys.q_dot;
    const int n = sys.total_dof;
    const Real h = 1e-6;
    auto mass_at = [&](const VecX& q) {
        sys.q = q;
        sys.compute_forward_kinematics();
        return mbd::compute_mass_matrix(sys);
    };
    const MatX Mdot = (mass_at(q0 + h * qd) - mass_at(q0 - h * qd)) / (2 * h);
    VecX bias = Mdot * qd;
    for (int k = 0; k < n; ++k) {
        VecX e = VecX::Zero(n);
        e(k) = h;
        const Real Tp = Real(0.5) * qd.dot(mass_at(q0 + e) * qd);
        const Real Tm = Real(0.5) * qd.dot(mass_at(q0 - e) * qd);
        bias(k) -= (Tp - Tm) / (2 * h);
    }
    sys.q = q0;
    sys.q_dot = qd;
    sys.compute_kinematics();
    return bias;
}

/// Gradient of the gravitational potential, by finite differences.
inline VecX potential_gradient(MultibodySystem& sys, const Vec3& gravity)
{
    const VecX q0 = sys.q;
    const int n = sys.total_dof;
    const Real h = 1e-6;
    VecX grad(n);
    for (int k = 0; k < n; ++k) {
        VecX e = VecX::Zero(n);
        e(k) = h;
        sys.q = q0 + e; sys.compute_forward_kinematics();
        const Real Vp = potential_energy(sys, gravity);
        sys.q = q0 - e; sys.compute_forward_kinematics();
        const Real Vm = potential_energy(sys, gravity);
        grad(k) = (Vp - Vm) / (2 * h);
    }
    sys.q = q0;
    sys.compute_kinematics();
    return grad;
}

struct DynamicsErrors {
    Real rnea_vs_mass{0.0};      ///< |(ID(qdd) - ID(0)) - M qdd|
    Real bias_vs_lagrange{0.0};  ///< |ID(0) - h_Lagrange|, no gravity
    Real gravity_vs_potential{0.0}; ///< |(ID_g(0) - ID_0(0)) - dV/dq|
    Real scale_mass{1.0};
    Real scale_bias{1.0};
    Real scale_gravity{1.0};
};

inline DynamicsErrors dynamics_errors(MultibodySystem& sys, Rng& rng)
{
    const int n = sys.total_dof;
    sys.compute_kinematics();
    const MatX M = mbd::compute_mass_matrix(sys);

    VecX qdd(n);
    for (int i = 0; i < n; ++i) qdd(i) = rng.range(-2.0, 2.0);

    const Vec3 no_gravity = Vec3::Zero();
    const Vec3 gravity(0.3, -9.81, 1.1);   // deliberately not axis-aligned

    const VecX tau_acc  = mbd::inverse_dynamics(sys, qdd, no_gravity);
    const VecX tau_zero = mbd::inverse_dynamics(sys, VecX::Zero(n), no_gravity);
    const VecX tau_grav = mbd::inverse_dynamics(sys, VecX::Zero(n), gravity);
    const VecX h_lagr   = lagrangian_bias(sys);
    const VecX dV_dq    = potential_gradient(sys, gravity);

    DynamicsErrors e;
    e.rnea_vs_mass         = ((tau_acc - tau_zero) - M * qdd).norm();
    e.bias_vs_lagrange     = (tau_zero - h_lagr).norm();
    e.gravity_vs_potential = ((tau_grav - tau_zero) - dV_dq).norm();
    e.scale_mass    = Real(1.0) + (M * qdd).norm();
    e.scale_bias    = Real(1.0) + h_lagr.norm();
    e.scale_gravity = Real(1.0) + dV_dq.norm();
    return e;
}

// ============================================================================
// Check 3: constraints against finite differences
// ============================================================================

struct ConstraintErrors {
    Real jacobian{0.0};      ///< |J q_dot - dPhi/dt|
    Real bias{0.0};          ///< |c - d2Phi/dt2| at zero joint acceleration
    Real scale_vel{1.0};
    Real scale_acc{1.0};
    Real rank_ratio{1.0};    ///< smallest / largest singular value of the rows
};

/// One entry per constraint in sys.constraints.
inline std::vector<ConstraintErrors> constraint_errors(MultibodySystem& sys)
{
    const VecX q0 = sys.q, qd = sys.q_dot;
    const int n = sys.total_dof;
    auto phi_at = [&](const VecX& q) {
        sys.q = q;
        sys.compute_forward_kinematics();
        return mbd::evaluate_all_constraints(sys);
    };
    const Real h = 1e-4;
    const VecX php = phi_at(q0 + h * qd);
    const VecX phm = phi_at(q0 - h * qd);
    const VecX ph0 = phi_at(q0);
    const VecX phid_num  = (php - phm) / (2 * h);
    const VecX phidd_num = (php - 2 * ph0 + phm) / (h * h);

    sys.q = q0;
    sys.q_dot = qd;
    sys.compute_kinematics();
    const MatX Jq = mbd::build_constraint_jacobian(sys);
    const VecX phid = Jq * qd;
    const auto acc = mbd::compute_body_accelerations(sys, VecX::Zero(n));

    std::vector<ConstraintErrors> out;
    int row = 0;
    for (const auto& c : sys.constraints) {
        const int ne = c->equation_count();
        Eigen::MatrixXd J1, J2;
        c->jacobian(sys, J1, J2);
        Eigen::VectorXd gamma;
        c->velocity_bias(sys, gamma);
        Vec6 a1, a2;
        a1 << acc[c->body1_idx].a, acc[c->body1_idx].alpha;
        a2 << acc[c->body2_idx].a, acc[c->body2_idx].alpha;
        const VecX c_bias = J1 * a1 + J2 * a2 + gamma;

        ConstraintErrors e;
        e.jacobian  = (phid.segment(row, ne) - phid_num.segment(row, ne)).norm();
        e.bias      = (c_bias - phidd_num.segment(row, ne)).norm();
        e.scale_vel = Real(1.0) + phid_num.segment(row, ne).norm();
        e.scale_acc = Real(1.0) + phidd_num.segment(row, ne).norm();

        Eigen::JacobiSVD<MatX> svd(Jq.block(row, 0, ne, n));
        const auto& sv = svd.singularValues();
        e.rank_ratio = (sv.size() > 0 && sv(0) > 0) ? sv(sv.size() - 1) / sv(0) : Real(0.0);

        out.push_back(e);
        row += ne;
    }
    return out;
}

/// Largest constraint violation over a run with every form of drift
/// correction switched off. If the acceleration-level equations are exact,
/// only the integrator's own error is left. The state must already satisfy the
/// constraints at position level; velocities are made consistent here.
inline Real drift_without_stabilisation(MultibodySystem& sys, const Vec3& gravity,
                                        Real duration, Real dt)
{
    sys.compute_kinematics();
    mbd::project_onto_constraints(sys);

    mbd::Simulator sim(sys);
    sim.set_gravity(gravity);
    sim.method = mbd::IntegrationMethod::RK4;
    sim.project_constraints = false;
    sim.constraint_alpha = 0.0;
    sim.constraint_beta = 0.0;
    sim.initialize();

    Real worst = 0.0;
    const int steps = static_cast<int>(std::round(duration / dt));
    for (int i = 0; i < steps; ++i) {
        sim.step(dt);
        worst = std::max(worst, mbd::evaluate_all_constraints(sys).norm());
    }
    return worst;
}

} // namespace mbd_test
