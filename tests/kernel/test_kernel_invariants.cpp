// Invariants of the kinematics kernel (plan task 2.4), for every joint model
// as a root and as a child, on chains and on a branching tree:
//
//   - body velocities and accelerations are time derivatives of the poses;
//   - the Jacobian maps joint velocities to body velocities;
//   - RNEA equals the virtual power of each body's Newton-Euler residual;
//   - RNEA, CRBA and ABA describe the same dynamics;
//   - gravity forces are the gradient of the potential energy;
//   - energy and momentum are conserved when they should be;
//   - a case with a closed-form solution (bead on a spinning rod).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mbd/kernel/algorithms.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::KJoint;
using mbd_test::Rng;
using mbd_test::all_kernel_joints;
using mbd_test::random_chain;
using mbd_test::random_configuration;
using mbd_test::random_tree;
using mbd_test::random_vector;
using mbd_test::rk4_step;

namespace {

std::string describe(const Model& model)
{
    std::string s;
    for (int i = 1; i < model.nbodies(); ++i) {
        s += (i > 1 ? " - " : "") + std::string(model.joint[i]->name())
           + "(" + std::to_string(model.parent[i]) + ")";
    }
    return s;
}

/// Every two-joint combination followed by a revolute, then trees.
std::vector<Model> test_models(Rng& rng)
{
    std::vector<Model> models;
    for (KJoint root : all_kernel_joints()) {
        for (KJoint child : all_kernel_joints()) {
            models.push_back(random_chain(rng, root, {child, KJoint::Revolute}));
        }
    }
    models.push_back(random_tree(rng, KJoint::Free, all_kernel_joints()));
    models.push_back(random_tree(rng, KJoint::Revolute,
                                 {KJoint::Spherical, KJoint::Universal, KJoint::Planar}));
    return models;
}

/// R_dot R^T = [w]x, from two rotations a time 2h apart.
Vec3 angular_velocity_fd(const Quat& R_plus, const Quat& R_minus, const Quat& R, Real h)
{
    const Mat3 Rdot = (R_plus.toRotationMatrix() - R_minus.toRotationMatrix()) / (2.0 * h);
    const Mat3 W = Rdot * R.toRotationMatrix().transpose();
    return 0.5 * Vec3(W(2, 1) - W(1, 2), W(0, 2) - W(2, 0), W(1, 0) - W(0, 1));
}

Real total_energy(const Model& model, Data& data, const VecX& q, const VecX& v)
{
    forward_kinematics(model, data, q, v);
    return kinetic_energy(model, data) + potential_energy(model, data);
}

} // namespace

TEST_CASE("Kernel: body velocities and accelerations are derivatives of the poses",
          "[kernel][invariants]")
{
    Rng rng(11);
    for (const Model& model : test_models(rng)) {
        INFO(describe(model));
        Data data(model), d_plus(model), d_minus(model);
        const VecX q = random_configuration(model, rng);
        const VecX v = random_vector(model.nv, rng, 1.5);
        const VecX a = random_vector(model.nv, rng, 1.5);
        forward_kinematics(model, data, q, v, a);

        // Move along the trajectory through (q, v) with v_dot = a.
        const Real h = 1e-6;
        VecX qd;
        q_dot(model, q, v, qd);
        forward_kinematics(model, d_plus, VecX(q + h * qd), VecX(v + h * a));
        forward_kinematics(model, d_minus, VecX(q - h * qd), VecX(v - h * a));

        MatX J;
        for (int i = 1; i < model.nbodies(); ++i) {
            INFO("body " << i);
            const Vec6 V = body_velocity_world(data, i);
            const Real scale = 1.0 + V.norm();

            const Vec3 w_fd = angular_velocity_fd(d_plus.oMi[i].q, d_minus.oMi[i].q, data.oMi[i].q, h);
            const Vec3 v_fd = (d_plus.oMi[i].p - d_minus.oMi[i].p) / (2.0 * h);
            CHECK((V.head<3>() - w_fd).norm() < 1e-7 * scale);
            CHECK((V.tail<3>() - v_fd).norm() < 1e-7 * scale);

            body_jacobian_world(model, data, i, J);
            CHECK((J * v - V).norm() < 1e-12 * scale);

            const Vec6 A    = body_acceleration_world(data, i);
            const Vec6 A_fd = (body_velocity_world(d_plus, i) - body_velocity_world(d_minus, i)) / (2.0 * h);
            CHECK((A - A_fd).norm() < 1e-6 * (1.0 + A.norm()));
        }
    }
}

TEST_CASE("Kernel: inverse dynamics is the virtual power of the body Newton-Euler residuals",
          "[kernel][invariants]")
{
    // tau = sum over bodies of J_c^T [n_c; f], with f = m (a_c - g) and
    // n_c = I_c alpha + w x I_c w about the centre of mass: d'Alembert's
    // principle, from the kinematics alone.
    Rng rng(12);
    for (const Model& model : test_models(rng)) {
        INFO(describe(model));
        Data data(model);
        const VecX q = random_configuration(model, rng);
        const VecX v = random_vector(model.nv, rng, 1.5);
        const VecX a = random_vector(model.nv, rng, 1.5);
        const VecX tau = rnea(model, data, q, v, a);

        forward_kinematics(model, data, q, v, a);
        VecX tau_ref = VecX::Zero(model.nv);
        MatX J;
        for (int i = 1; i < model.nbodies(); ++i) {
            const RigidBodyInertia& body = model.inertia[i];
            const Mat3 R   = data.oMi[i].rotation_matrix();
            const Vec3 r   = R * body.com_B;                 // origin to centre of mass
            const Mat3 I_c = R * body.I_com_B * R.transpose();
            const Vec6 V   = body_velocity_world(data, i);
            const Vec6 A   = body_acceleration_world(data, i);
            const Vec3 w = V.head<3>(), alpha = A.head<3>();
            const Vec3 a_c = A.tail<3>() + alpha.cross(r) + w.cross(w.cross(r));

            const Vec3 f   = body.mass * (a_c - model.gravity);
            const Vec3 n_c = I_c * alpha + w.cross(I_c * w);

            body_jacobian_world(model, data, i, J);
            const MatX J_ang = J.topRows(3);
            const MatX J_com = J.bottomRows(3) - skew(r) * J_ang;
            tau_ref += J_ang.transpose() * n_c + J_com.transpose() * f;
        }
        CHECK((tau - tau_ref).norm() < 1e-10 * (1.0 + tau.norm()));
    }
}

TEST_CASE("Kernel: RNEA, CRBA and ABA describe the same dynamics", "[kernel][invariants]")
{
    Rng rng(13);
    for (const Model& model : test_models(rng)) {
        INFO(describe(model));
        Data data(model);
        const VecX q = random_configuration(model, rng);
        const VecX v = random_vector(model.nv, rng, 1.5);
        const VecX a = random_vector(model.nv, rng, 1.5);

        const MatX M = crba(model, data, q);
        CHECK((M - M.transpose()).norm() < 1e-14 * M.norm());
        CHECK(Eigen::LLT<MatX>(M).info() == Eigen::Success);

        const VecX bias = rnea(model, data, q, v, VecX::Zero(model.nv));
        const VecX tau  = rnea(model, data, q, v, a);
        const Real scale = 1.0 + tau.norm();
        CHECK((tau - bias - M * a).norm() < 1e-10 * scale);

        const VecX ddq = aba(model, data, q, v, tau);
        CHECK((ddq - a).norm() < 1e-9 * (1.0 + a.norm()));

        forward_kinematics(model, data, q, v);
        CHECK(std::abs(kinetic_energy(model, data) - 0.5 * v.dot(M * v)) < 1e-12 * (1.0 + v.dot(M * v)));
    }
}

TEST_CASE("Kernel: gravity forces are the gradient of the potential energy", "[kernel][invariants]")
{
    Rng rng(14);
    for (const Model& model : test_models(rng)) {
        INFO(describe(model));
        Data data(model);
        const VecX q = random_configuration(model, rng);
        const VecX g = rnea(model, data, q, VecX::Zero(model.nv), VecX::Zero(model.nv));

        // The generalized force of velocity k is dV/dq along G(q) e_k.
        const Real h = 1e-6;
        VecX grad(model.nv), dq;
        for (int k = 0; k < model.nv; ++k) {
            q_dot(model, q, VecX::Unit(model.nv, k), dq);
            forward_kinematics(model, data, VecX(q + h * dq));
            const Real V_plus = potential_energy(model, data);
            forward_kinematics(model, data, VecX(q - h * dq));
            const Real V_minus = potential_energy(model, data);
            grad(k) = (V_plus - V_minus) / (2.0 * h);
        }
        CHECK((g - grad).norm() < 1e-7 * (1.0 + g.norm()));
    }
}

TEST_CASE("Kernel: energy is conserved without applied forces", "[kernel][invariants]")
{
    Rng rng(15);
    std::vector<Model> models;
    models.push_back(random_tree(rng, KJoint::Free, all_kernel_joints()));
    models.push_back(random_tree(rng, KJoint::Revolute,
                                 {KJoint::Spherical, KJoint::Universal, KJoint::Planar, KJoint::Cylindrical}));
    models.push_back(random_chain(rng, KJoint::Spherical, {KJoint::Spherical, KJoint::Spherical}));

    for (const Model& model : models) {
        INFO(describe(model));
        Data data(model);
        VecX q = random_configuration(model, rng);
        VecX v = random_vector(model.nv, rng, 1.0);
        const Real E0 = total_energy(model, data, q, v);
        for (int step = 0; step < 1000; ++step) rk4_step(model, data, q, v, 1e-3);
        const Real E1 = total_energy(model, data, q, v);
        CHECK(std::abs(E1 - E0) < 1e-7 * (1.0 + std::abs(E0)));
    }
}

TEST_CASE("Kernel: a floating system conserves momentum", "[kernel][invariants]")
{
    Rng rng(16);
    Model model = random_tree(rng, KJoint::Free, all_kernel_joints());
    model.gravity.setZero();
    INFO(describe(model));
    Data data(model);
    VecX q = random_configuration(model, rng);
    VecX v = random_vector(model.nv, rng, 1.0);

    forward_kinematics(model, data, q, v);
    const Vec6 h0 = momentum_world(model, data);
    const Vec3 c0 = center_of_mass(model, data);
    const Real T = 1.0, dt = 1e-3;
    for (int step = 0; step < static_cast<int>(T / dt + 0.5); ++step) rk4_step(model, data, q, v, dt);
    forward_kinematics(model, data, q, v);
    const Vec6 h1 = momentum_world(model, data);
    CHECK((h1 - h0).norm() < 1e-9 * (1.0 + h0.norm()));

    // The centre of mass moves in a straight line at the speed P / m.
    Real mass = 0.0;
    for (int i = 1; i < model.nbodies(); ++i) mass += model.inertia[i].mass;
    const Vec3 c1 = center_of_mass(model, data);
    CHECK((c1 - c0 - h0.tail<3>() / mass * T).norm() < 1e-9);
}

TEST_CASE("Kernel: bead on a spinning rod follows r = r0 cosh(w t)", "[kernel][invariants]")
{
    // A horizontal rod turns about the vertical Z axis at rate w; a bead slides
    // freely along it. Radially m r'' = m w^2 r, so from rest r = r0 cosh(w t).
    // The rod's inertia is so large that the bead does not slow it measurably.
    Model model;
    const RigidBodyInertia rod(1e9, Vec3::Zero(), 1e9 * Mat3::Identity());
    const int rod_body = model.add_body(0, std::make_shared<RevoluteJointModel>(),
                                        Transform3::Identity(), Transform3::Identity(), rod, "rod");
    // The prismatic joint slides along its Z axis: turn it onto the rod's X.
    const Transform3 along_rod(Quat(Eigen::AngleAxisd(0.5 * pi, Vec3::UnitY())), Vec3::Zero());
    const RigidBodyInertia bead(0.2, Vec3::Zero(), 1e-4 * Mat3::Identity());
    model.add_body(rod_body, std::make_shared<PrismaticJointModel>(), along_rod,
                   Transform3::Identity(), bead, "bead");

    const Real w = 2.0, r0 = 0.1, T = 1.0, dt = 1e-3;
    Data data(model);
    VecX q(2), v(2);
    q << 0.0, r0;
    v << w, 0.0;
    for (int step = 0; step < static_cast<int>(T / dt + 0.5); ++step) rk4_step(model, data, q, v, dt);

    CHECK(std::abs(q(1) - r0 * std::cosh(w * T)) < 1e-9);
    CHECK(std::abs(v(1) - r0 * w * std::sinh(w * T)) < 1e-9);
    CHECK(std::abs(q(0) - w * T) < 1e-8);

    // The bead is where the kinematics say: on the rod, at radius r.
    forward_kinematics(model, data, q);
    const Vec3 p = data.oMi[2].p;
    CHECK(std::abs(p.head<2>().norm() - q(1)) < 1e-12);
    CHECK(std::abs(p.z()) < 1e-12);
}

TEST_CASE("Kernel: integrate, difference and q_dot agree", "[kernel][invariants]")
{
    Rng rng(17);
    const Model model = random_tree(rng, KJoint::Free, all_kernel_joints());
    INFO(describe(model));
    const VecX q = random_configuration(model, rng);
    const VecX v = random_vector(model.nv, rng, 2.0);

    // Quaternions stay unit, and integrating in place gives the same result.
    VecX q1;
    integrate(model, q, v, 0.3, q1);
    for (int i = 1; i < model.nbodies(); ++i) {
        const std::string name = model.joint[i]->name();
        if (name == "spherical") CHECK(std::abs(q1.segment<4>(model.idx_q[i]).norm() - 1.0) < 1e-15);
        if (name == "free") CHECK(std::abs(q1.segment<4>(model.idx_q[i] + 3).norm() - 1.0) < 1e-15);
    }
    VecX q2 = q;
    integrate(model, q2, v, 0.3, q2);
    CHECK(q2 == q1);

    // difference inverts integrate.
    VecX dv;
    difference(model, q, q1, dv);
    CHECK((dv - 0.3 * v).norm() < 1e-13);
    const VecX q_other = random_configuration(model, rng);
    difference(model, q, q_other, dv);
    integrate(model, q, dv, 1.0, q2);
    VecX residual;
    difference(model, q_other, q2, residual);
    CHECK(residual.norm() < 1e-12);

    // The rate of integrate is q_dot.
    VecX qd;
    q_dot(model, q, v, qd);
    const Real h = 1e-6;
    VecX q_plus, q_minus;
    integrate(model, q, v, h, q_plus);
    integrate(model, q, v, -h, q_minus);
    CHECK(((q_plus - q_minus) / (2.0 * h) - qd).norm() < 1e-8);

    // integrate is exact for a constant velocity: compare with RK4 on
    // q_dot = G(q) v with v held fixed.
    VecX q_rk = q, k1, k2, k3, k4;
    const int steps = 1000;
    const Real dt = 1.0 / steps;
    for (int step = 0; step < steps; ++step) {
        q_dot(model, q_rk, v, k1);
        q_dot(model, VecX(q_rk + 0.5 * dt * k1), v, k2);
        q_dot(model, VecX(q_rk + 0.5 * dt * k2), v, k3);
        q_dot(model, VecX(q_rk + dt * k3), v, k4);
        q_rk += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        normalize(model, q_rk);
    }
    integrate(model, q, v, 1.0, q1);
    difference(model, q1, q_rk, residual);
    CHECK(residual.norm() < 1e-10);

    // Neutral configuration: the joint frames coincide.
    Data data(model);
    forward_kinematics(model, data, model.neutral_configuration());
    for (int i = 1; i < model.nbodies(); ++i) {
        const Transform3 expected = data.oMi[model.parent[i]] * model.X_PJ[i] * model.X_JC[i];
        CHECK((data.oMi[i].p - expected.p).norm() < 1e-14);
        CHECK(data.oMi[i].q.angularDistance(expected.q) < 1e-12);
    }
}
