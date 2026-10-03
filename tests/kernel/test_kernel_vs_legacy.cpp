// The kernel against the legacy algorithms (plan task 2.4). The same model is
// built in both and compared on poses, velocities, the mass matrix and the
// accelerations of forward dynamics. Removed with the legacy path (task 2.7).
//
// The two use different coordinates for spherical and free joints: the legacy
// path a rotation vector r with velocities r_dot (and t_dot for the free
// joint, in the parent-side joint frame), the kernel a quaternion with the
// angular velocity w = E(r) r_dot and the linear velocity R(r)^T t_dot, both
// in the child-side joint frame. With qd_legacy = T v_kernel, the mass
// matrices are related by M_kernel = T^T M_legacy T and the generalized
// forces by tau_kernel = T^T tau_legacy.

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "mbd/algorithms/dynamics.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/model/system.hpp"

#include "invariants/invariant_helpers.hpp"
#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using mbd_test::JointKind;
using mbd_test::KJoint;
using mbd_test::Rng;

namespace {

const Vec3 kGravity(0.0, 0.0, -g_accel);

KJoint kernel_kind(JointKind k)
{
    switch (k) {
        case JointKind::Revolute:  return KJoint::Revolute;
        case JointKind::Prismatic: return KJoint::Prismatic;
        case JointKind::Spherical: return KJoint::Spherical;
        case JointKind::Universal: return KJoint::Universal;
        case JointKind::Free:      return KJoint::Free;
        case JointKind::Fixed:     return KJoint::Fixed;
    }
    return KJoint::Fixed;
}

/// One model, built in both engines.
struct Twin {
    MultibodySystem sys;
    kernel::Model model;
    std::vector<JointKind> kind{JointKind::Fixed};  // per body; [0] is ground

    void add_body(Rng& rng, JointKind k, int parent)
    {
        const RigidBodyInertia inertia = mbd_test::random_inertia(rng);
        const Transform3 X_PJ = rng.frame(0.4);
        const Transform3 X_CJ = rng.frame(0.3);
        const BodyIndex b = sys.add_body(inertia, RigidBodyState{}, "", parent);
        sys.add_joint(mbd_test::make_joint(k, X_PJ, X_CJ, parent, b));
        model.add_body(parent, mbd_test::make_kernel_joint(kernel_kind(k)), X_PJ, X_CJ, inertia);
        kind.push_back(k);
    }

    std::string describe() const
    {
        std::string s;
        for (std::size_t i = 1; i < kind.size(); ++i) {
            s += (i > 1 ? " - " : "") + std::string(mbd_test::joint_name(kind[i]));
        }
        return s;
    }

    /// The legacy state (sys.q, sys.q_dot) in kernel coordinates, and T.
    void kernel_state(VecX& q, VecX& v, MatX& T) const
    {
        q.resize(model.nq);
        v.resize(model.nv);
        T = MatX::Zero(model.nv, model.nv);
        for (int i = 1; i < model.nbodies(); ++i) {
            const Joint& joint = *sys.joints[static_cast<std::size_t>(sys.body_infos[i].joint_idx)];
            const int o = joint.q_offset, n = joint.num_dof();
            const int iq = model.idx_q[i], iv = model.idx_v[i];
            REQUIRE(iv == o);
            const VecX qo  = sys.q.segment(o, n);
            const VecX qdo = sys.q_dot.segment(o, n);
            switch (kind[static_cast<std::size_t>(i)]) {
                case JointKind::Spherical: {
                    const Vec3 r = qo.head<3>();
                    const Mat3 E = detail::exp_map_jacobian(r);
                    q.segment<4>(iq) = Quat(detail::exp_map_rotation(r)).coeffs();
                    v.segment<3>(iv) = E * qdo.head<3>();
                    T.block<3, 3>(o, iv) = E.inverse();
                    break;
                }
                case JointKind::Free: {
                    const Vec3 r = qo.tail<3>();
                    const Mat3 R = detail::exp_map_rotation(r);
                    const Mat3 E = detail::exp_map_jacobian(r);
                    q.segment<3>(iq)     = qo.head<3>();
                    q.segment<4>(iq + 3) = Quat(R).coeffs();
                    v.segment<3>(iv)     = E * qdo.tail<3>();
                    v.segment<3>(iv + 3) = R.transpose() * qdo.head<3>();
                    T.block<3, 3>(o, iv + 3) = R;            // t_dot = R v_linear
                    T.block<3, 3>(o + 3, iv) = E.inverse();  // r_dot = E^-1 w
                    break;
                }
                default:
                    q.segment(iq, n) = qo;
                    v.segment(iv, n) = qdo;
                    T.block(o, iv, n, n).setIdentity();
                    break;
            }
        }
    }
};

/// Chains of every pair of legacy joint kinds, with a revolute after them and
/// a universal branch off the root.
std::vector<std::pair<JointKind, JointKind>> joint_pairs()
{
    const std::vector<JointKind> kinds{JointKind::Revolute, JointKind::Prismatic, JointKind::Spherical,
                                       JointKind::Universal, JointKind::Free, JointKind::Fixed};
    std::vector<std::pair<JointKind, JointKind>> pairs;
    for (JointKind a : kinds) {
        for (JointKind b : kinds) pairs.emplace_back(a, b);
    }
    return pairs;
}

void build(Twin& twin, Rng& rng, JointKind root, JointKind child)
{
    twin.add_body(rng, root, 0);
    twin.add_body(rng, child, 1);
    twin.add_body(rng, JointKind::Revolute, 2);
    twin.add_body(rng, JointKind::Universal, 1);
    twin.model.gravity = kGravity;
    for (int k = 0; k < twin.sys.total_dof; ++k) {
        twin.sys.q(k)     = rng.range(-0.6, 0.6);
        twin.sys.q_dot(k) = rng.range(-1.2, 1.2);
    }
    twin.sys.compute_kinematics();
}

} // namespace

TEST_CASE("Kernel vs legacy: poses and velocities", "[kernel][legacy]")
{
    Rng rng(21);
    for (const auto& [root, child] : joint_pairs()) {
        Twin twin;
        build(twin, rng, root, child);
        INFO(twin.describe());

        VecX q, v;
        MatX T;
        twin.kernel_state(q, v, T);
        kernel::Data data(twin.model);
        kernel::forward_kinematics(twin.model, data, q, v);

        for (int i = 1; i < twin.model.nbodies(); ++i) {
            INFO("body " << i);
            const RigidBodyState& s = twin.sys.states[static_cast<std::size_t>(i)];
            CHECK((data.oMi[i].p - s.p_WB).norm() < 1e-12);
            CHECK(data.oMi[i].q.angularDistance(s.q_WB) < 1e-12);
            const Vec6 V = kernel::body_velocity_world(data, i);
            CHECK((V.head<3>() - s.w_WB).norm() < 1e-12 * (1.0 + V.norm()));
            CHECK((V.tail<3>() - s.v_WB).norm() < 1e-12 * (1.0 + V.norm()));
        }
    }
}

TEST_CASE("Kernel vs legacy: mass matrix and kinetic energy", "[kernel][legacy]")
{
    Rng rng(22);
    for (const auto& [root, child] : joint_pairs()) {
        Twin twin;
        build(twin, rng, root, child);
        INFO(twin.describe());

        VecX q, v;
        MatX T;
        twin.kernel_state(q, v, T);
        kernel::Data data(twin.model);

        const MatX M_legacy = compute_mass_matrix(twin.sys);
        const MatX M = kernel::crba(twin.model, data, q);
        CHECK((M - T.transpose() * M_legacy * T).norm() < 1e-10 * (1.0 + M.norm()));

        kernel::forward_kinematics(twin.model, data, q, v);
        const Real T_legacy = 0.5 * twin.sys.q_dot.dot(M_legacy * twin.sys.q_dot);
        CHECK(std::abs(kernel::kinetic_energy(twin.model, data) - T_legacy) < 1e-10 * (1.0 + T_legacy));
        CHECK(std::abs(kernel::potential_energy(twin.model, data)
                       - mbd_test::potential_energy(twin.sys, kGravity)) < 1e-10);
    }
}

TEST_CASE("Kernel vs legacy: accelerations of forward dynamics", "[kernel][legacy]")
{
    Rng rng(23);
    for (const auto& [root, child] : joint_pairs()) {
        Twin twin;
        build(twin, rng, root, child);
        INFO(twin.describe());

        VecX q, v;
        MatX T;
        twin.kernel_state(q, v, T);
        kernel::Data data(twin.model);

        VecX tau_legacy(twin.sys.total_dof);
        for (int k = 0; k < tau_legacy.size(); ++k) tau_legacy(k) = rng.range(-5.0, 5.0);
        const VecX qdd_legacy = forward_dynamics(twin.sys, tau_legacy, kGravity);
        const auto acc_legacy = compute_body_accelerations(twin.sys, qdd_legacy);

        kernel::aba(twin.model, data, q, v, VecX(T.transpose() * tau_legacy));

        for (int i = 1; i < twin.model.nbodies(); ++i) {
            INFO("body " << i);
            const Vec6 A = kernel::body_acceleration_world(data, i);
            const auto& a = acc_legacy[static_cast<std::size_t>(i)];
            CHECK((A.head<3>() - a.alpha).norm() < 1e-10 * (1.0 + A.norm()));
            CHECK((A.tail<3>() - a.a).norm() < 1e-10 * (1.0 + A.norm()));
        }
    }
}
