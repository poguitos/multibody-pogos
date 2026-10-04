// Constraint primitives and closures of the kernel (plan task 2.5).
//
//   - J and nu are the first time derivative of phi, gamma the second
//     (finite differences along a trajectory, with time-dependent targets);
//   - the closures are satisfied when assembled, have full rank, and allow
//     exactly the relative motions of the joint they stand for.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::KJoint;
using mbd_test::Rng;

namespace {

struct Evaluation {
    VecX phi, nu, gamma;
    MatX J;
};

Evaluation evaluate(const Model& model, const ConstraintModel& c,
                    const VecX& q, const VecX& v, Real t)
{
    Data data(model);
    forward_kinematics(model, data, q, v, VecX::Zero(model.nv));
    Evaluation e;
    const int m = c.size();
    e.phi.resize(m);
    e.nu.resize(m);
    e.gamma.resize(m);
    e.J.resize(m, model.nv);
    c.calc(model, data, t, e.phi, e.J, e.nu, e.gamma);
    return e;
}

TimeFunction wave(Real offset, Real amplitude, Real omega)
{
    return TimeFunction([=](Real t) { return offset + amplitude * std::sin(omega * t); },
                        [=](Real t) { return amplitude * omega * std::cos(omega * t); },
                        [=](Real t) { return -amplitude * omega * omega * std::sin(omega * t); });
}

/// Every primitive, with constant and time-dependent targets, and every
/// closure, between markers i and j.
std::vector<std::shared_ptr<const ConstraintModel>> all_constraints(const Marker& i, const Marker& j)
{
    return {
        std::make_shared<PointCoincidence>(i, j),
        std::make_shared<Dot1>(i, 2, j, 0),
        std::make_shared<Dot1>(i, 1, j, 2, wave(0.2, 0.3, 2.0)),
        std::make_shared<Dot2>(i, 0, j),
        std::make_shared<Dot2>(i, 2, j, wave(-0.1, 0.2, 3.0)),
        std::make_shared<Distance>(i, j, 0.7),
        std::make_shared<Distance>(i, j, wave(0.6, 0.1, 1.5), 0.6),
        std::make_shared<NoTwist>(i, j),
        point_on_line(i, j),
        point_on_plane(i, j),
        spherical_closure(i, j),
        revolute_closure(i, j),
        universal_closure(i, j),
        cylindrical_closure(i, j),
        prismatic_closure(i, j),
        constant_velocity_closure(i, j),
    };
}

/// Rank by singular values, relative to the largest.
int rank_of(const MatX& J)
{
    const Eigen::JacobiSVD<MatX> svd(J);
    const VecX s = svd.singularValues();
    int r = 0;
    for (Index k = 0; k < s.size(); ++k) {
        if (s(k) > 1e-9 * s(0)) ++r;
    }
    return r;
}

} // namespace

TEST_CASE("Kernel constraints: J, nu and gamma are derivatives of phi", "[kernel][constraints]")
{
    Rng rng(41);
    // Bodies 1..5: a free root, 2 and 3 on one branch, 4 and 5 on another.
    const Model model = mbd_test::random_tree(rng, KJoint::Free, mbd_test::all_kernel_joints());
    // Different branches, child and parent, ground and a body (both ways),
    // and two markers on one body.
    const std::vector<std::pair<int, int>> pairs{{3, 5}, {3, 2}, {0, 5}, {4, 0}, {5, 5}};

    for (const auto& [bi, bj] : pairs) {
        const Marker mi{bi, rng.frame(0.3)};
        const Marker mj{bj, rng.frame(0.3)};
        for (const auto& c : all_constraints(mi, mj)) {
            INFO(c->name() << ", bodies " << bi << " and " << bj);
            const VecX q = mbd_test::random_configuration(model, rng);
            const VecX v = mbd_test::random_vector(model.nv, rng, 1.5);
            const VecX a = mbd_test::random_vector(model.nv, rng, 1.5);
            const Real t = rng.range(0.0, 2.0);
            const Evaluation e = evaluate(model, *c, q, v, t);

            // Along the trajectory through (q, v, t) with v_dot = a.
            const Real h = 1e-6;
            VecX qd;
            q_dot(model, q, v, qd);
            const VecX v_plus = v + h * a, v_minus = v - h * a;
            const Evaluation ep = evaluate(model, *c, VecX(q + h * qd), v_plus, t + h);
            const Evaluation em = evaluate(model, *c, VecX(q - h * qd), v_minus, t - h);

            // d(phi)/dt = J v - nu.
            const VecX phi_dot = (ep.phi - em.phi) / (2.0 * h);
            CHECK((phi_dot - (e.J * v - e.nu)).norm() < 1e-7 * (1.0 + phi_dot.norm()));

            // d2(phi)/dt2 = d/dt (J v - nu) = J a - gamma.
            const VecX phi_ddot = ((ep.J * v_plus - ep.nu) - (em.J * v_minus - em.nu)) / (2.0 * h);
            CHECK((phi_ddot - (e.J * a - e.gamma)).norm() < 1e-6 * (1.0 + phi_ddot.norm()));
        }
    }
}

TEST_CASE("Kernel constraints: closures allow exactly the motions of their joint",
          "[kernel][constraints]")
{
    // A free body and a ground marker. The body's marker is placed to coincide
    // with the ground marker (for the universal joint, turned so that the two
    // Z axes are the perpendicular arms of the cross). The velocities that
    // satisfy J v = 0 must be the joint's motions, no more and no fewer.
    Rng rng(42);
    Model model;
    model.add_body(0, std::make_shared<FreeJointModel>(), Transform3::Identity(),
                   Transform3::Identity(), mbd_test::random_inertia(rng));
    const VecX q = mbd_test::random_configuration(model, rng);
    Data data(model);
    forward_kinematics(model, data, q, VecX::Zero(6), VecX::Zero(6));

    const Marker ground{0, rng.frame(1.0)};
    const Transform3 X_BM = data.oMi[1].inverse() * ground.X_BM;   // coincident
    const Marker body{1, X_BM};
    const Marker body_turned{1, X_BM * Transform3(Quat(Eigen::AngleAxisd(0.5 * pi, Vec3::UnitX())),
                                                  Vec3::Zero())};

    const Mat3 R_W = data.oMi[1].rotation_matrix();     // body axes in world
    const Vec3 r_B = X_BM.p;                             // marker origin in body
    const Vec3 z = ground.X_BM.rotation_matrix().col(2); // joint axis in world

    struct Case {
        std::shared_ptr<ConstraintSet> c;
        int dof;
        // Checks a world angular velocity w and marker-point velocity vp.
        std::function<bool(const Vec3& w, const Vec3& vp)> allowed;
    };
    const Real tol = 1e-10;
    const Vec3 z_turned = R_W * body_turned.X_BM.rotation_matrix().col(2);
    const std::vector<Case> cases{
        {spherical_closure(ground, body), 3,
         [&](const Vec3&, const Vec3& vp) { return vp.norm() < tol; }},
        {revolute_closure(ground, body), 1,
         [&](const Vec3& w, const Vec3& vp) { return vp.norm() < tol && w.cross(z).norm() < tol; }},
        {universal_closure(ground, body_turned), 2,
         [&](const Vec3& w, const Vec3& vp) {
             return vp.norm() < tol && std::abs(w.dot(z.cross(z_turned))) < tol; }},
        {cylindrical_closure(ground, body), 2,
         [&](const Vec3& w, const Vec3& vp) { return vp.cross(z).norm() < tol && w.cross(z).norm() < tol; }},
        {prismatic_closure(ground, body), 1,
         [&](const Vec3& w, const Vec3& vp) { return vp.cross(z).norm() < tol && w.norm() < tol; }},
        {constant_velocity_closure(ground, body), 2,
         [&](const Vec3& w, const Vec3& vp) { return vp.norm() < tol && std::abs(w.dot(z)) < tol; }},
    };

    for (const Case& k : cases) {
        INFO(k.c->name());
        const Evaluation e = evaluate(model, *k.c, q, VecX::Zero(6), 0.0);
        CHECK(e.phi.norm() < 1e-12);
        CHECK(rank_of(e.J) == k.c->size());
        CHECK(6 - k.c->size() == k.dof);

        // Every velocity in the null space of J is a motion of the joint.
        const Eigen::FullPivLU<MatX> lu(e.J);
        const MatX N = lu.kernel();
        REQUIRE(N.cols() == k.dof);
        for (Index n = 0; n < N.cols(); ++n) {
            const Vec3 w_B = N.col(n).head<3>(), v_B = N.col(n).tail<3>();
            const Vec3 w  = R_W * w_B;
            const Vec3 vp = R_W * (v_B + w_B.cross(r_B));
            CHECK(k.allowed(w / N.col(n).norm(), vp / N.col(n).norm()));
        }
    }
}
