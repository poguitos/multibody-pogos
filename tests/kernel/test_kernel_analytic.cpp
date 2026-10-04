// Closed-form cases on the kernel (plan task 2.7). They replace the tests of
// the single-body integrator (dynamics_og.hpp) and of the body-level
// constraint solver (solvers/solver.hpp), both deleted:
//
//   - a free body moves at constant velocity, falls freely, and turns as
//     Euler's equations say, including the torque-free symmetric top;
//   - a pendulum held by a distance constraint pulls with m g and
//     m g + m v^2 / L, and swings with the period of the elliptic integral;
//   - a bar released horizontally from a hinge loads it with
//     m g (1 - m (L/2)^2 / I_hinge), a quarter of its weight for a thin rod.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/spatial/spatial.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;

namespace {

/// One body on a free joint whose frames are the body frame and the world,
/// so v = (w, v_origin) in body axes.
Model free_body(const RigidBodyInertia& inertia, const Vec3& gravity)
{
    Model model;
    model.gravity = gravity;
    model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), inertia);
    return model;
}

VecX free_configuration(const Vec3& p, const Quat& R)
{
    VecX q(7);
    q << p, R.coeffs();
    return q;
}

/// One RK4 step of a constrained system, then projection onto the
/// constraints.
void rk4_constrained(const Model& model, ConstraintSolver& solver, Data& data,
                     VecX& q, VecX& v, Real t, Real dt)
{
    const VecX zero = VecX::Zero(model.nv);
    VecX k1q, k2q, k3q, k4q;
    q_dot(model, q, v, k1q);
    const VecX k1v = solver.forward_dynamics(data, q, v, zero, t);
    const VecX q2 = q + 0.5 * dt * k1q, v2 = v + 0.5 * dt * k1v;
    q_dot(model, q2, v2, k2q);
    const VecX k2v = solver.forward_dynamics(data, q2, v2, zero, t + 0.5 * dt);
    const VecX q3 = q + 0.5 * dt * k2q, v3 = v + 0.5 * dt * k2v;
    q_dot(model, q3, v3, k3q);
    const VecX k3v = solver.forward_dynamics(data, q3, v3, zero, t + 0.5 * dt);
    const VecX q4 = q + dt * k3q, v4 = v + dt * k3v;
    q_dot(model, q4, v4, k4q);
    const VecX k4v = solver.forward_dynamics(data, q4, v4, zero, t + dt);
    q += dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
    v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    normalize(model, q);
    solver.project(data, q, v, t + dt);
}

/// Complete elliptic integral of the first kind, K(k), by the
/// arithmetic-geometric mean: K = pi / (2 AGM(1, sqrt(1 - k^2))).
Real elliptic_K(Real k)
{
    Real a = 1.0, b = std::sqrt(1.0 - k * k);
    for (int i = 0; i < 30 && std::abs(a - b) > 1e-16; ++i) {
        const Real m = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = m;
    }
    return pi / (2.0 * a);
}

} // namespace

// --- One free body -------------------------------------------------------------

TEST_CASE("Kernel analytic: a free body without forces moves at constant velocity",
          "[kernel][analytic]")
{
    // A cube: every axis is principal, so a constant body angular velocity
    // solves Euler's equations and the attitude is R0 exp(w t). Its centre
    // of mass, the body origin, moves in a straight line.
    const Model model = free_body(RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.5, 0.5)),
                                  Vec3::Zero());
    Data data(model);
    const Quat R0(Eigen::AngleAxisd(0.4, Vec3(1.0, 2.0, -1.0).normalized()));
    VecX q = free_configuration(Vec3(0.1, -0.2, 0.3), R0);
    VecX v(6);
    v << 0.3, -0.2, 0.5, 1.0, -2.0, 0.5;
    const Vec3 w = v.head<3>();
    const Vec3 velocity_world = R0 * v.tail<3>();

    const Real dt = 0.01, T = 1.0;
    for (int step = 0; step < 100; ++step) mbd_test::rk4_step(model, data, q, v, dt);

    forward_kinematics(model, data, q, v);
    // RK4 turns the attitude and the body-frame velocity by slightly different
    // polynomials of w dt, so the world velocity drifts by O((w dt)^5) a step.
    CHECK((body_velocity_world(data, 1).tail<3>() - velocity_world).norm() < 1e-10);
    CHECK((data.oMi[1].p - (Vec3(0.1, -0.2, 0.3) + velocity_world * T)).norm() < 1e-10);
    CHECK(data.oMi[1].q.angularDistance(R0 * exp3(w * T)) < 1e-10);
}

TEST_CASE("Kernel analytic: a free body falls freely", "[kernel][analytic]")
{
    // Constant acceleration: RK4 is exact for it.
    const Model model = free_body(RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.5, 0.5)),
                                  Vec3(0.0, 0.0, -g_accel));
    Data data(model);
    VecX q = free_configuration(Vec3::Zero(), Quat::Identity());
    VecX v = VecX::Zero(6);
    for (int step = 0; step < 100; ++step) mbd_test::rk4_step(model, data, q, v, 0.01);

    forward_kinematics(model, data, q, v);
    CHECK((data.oMi[1].p - Vec3(0.0, 0.0, -0.5 * g_accel)).norm() < 1e-12);
    CHECK((body_velocity_world(data, 1) - (Vec6() << 0, 0, 0, 0, 0, -g_accel).finished()).norm() < 1e-12);
}

TEST_CASE("Kernel analytic: Euler's equations give the gyroscopic acceleration",
          "[kernel][analytic]")
{
    // I w_dot = -w x (I w), with principal moments (Ixx, Iyy, Izz) and
    // w = (1, 1, 1): w_dot = (-(Izz - Iyy)/Ixx, -(Ixx - Izz)/Iyy, -(Iyy - Ixx)/Izz).
    const RigidBodyInertia body = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.2, 0.3));
    const Model model = free_body(body, Vec3::Zero());
    Data data(model);
    const VecX q = free_configuration(Vec3::Zero(), Quat::Identity());
    VecX v = VecX::Zero(6);
    v.head<3>() = Vec3(1.0, 1.0, 1.0);

    const VecX a = aba(model, data, q, v, VecX::Zero(6));
    const Real Ixx = body.I_com_B(0, 0), Iyy = body.I_com_B(1, 1), Izz = body.I_com_B(2, 2);
    const Vec3 expected(-(Izz - Iyy) / Ixx, -(Ixx - Izz) / Iyy, -(Iyy - Ixx) / Izz);
    CHECK((a.head<3>() - expected).norm() < 1e-12);
    CHECK(a.tail<3>().norm() < 1e-12);   // the centre of mass, at the origin, stays put
}

TEST_CASE("Kernel analytic: the torque-free symmetric top", "[kernel][analytic]")
{
    // I1 = I2: w3 stays constant and (w1, w2) turns at the rate
    // Omega = (I3 - I1) w3 / I1 in the body frame.
    const RigidBodyInertia body = RigidBodyInertia::from_solid_box(2.0, Vec3(0.3, 0.3, 0.1));
    const Model model = free_body(body, Vec3::Zero());
    Data data(model);
    VecX q = free_configuration(Vec3::Zero(), Quat(Eigen::AngleAxisd(0.3, Vec3::UnitX())));
    VecX v = VecX::Zero(6);
    const Real w1 = 0.7, w3 = 3.0;
    v.head<3>() = Vec3(w1, 0.0, w3);

    const Real I1 = body.I_com_B(0, 0), I3 = body.I_com_B(2, 2);
    const Real Omega = (I3 - I1) * w3 / I1;
    const Real dt = 1e-3, T = 2.0;
    for (int step = 0; step < 2000; ++step) mbd_test::rk4_step(model, data, q, v, dt);

    const Vec3 expected(w1 * std::cos(Omega * T), w1 * std::sin(Omega * T), w3);
    CHECK((v.head<3>() - expected).norm() < 1e-9);
}

// --- Constrained, against closed forms ------------------------------------------

TEST_CASE("Kernel analytic: a pendulum rod pulls with m g and m g + m v^2 / L",
          "[kernel][analytic]")
{
    // A bob of 1 kg at the end of a 1 m distance constraint to the pivot at
    // the origin. The constraint acts at the bob's centre of mass, so the
    // force J^T lambda on it has no moment.
    const Real m = 1.0, L = 1.0;
    const Model model = free_body(RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.1)),
                                  Vec3(0.0, 0.0, -g_accel));
    Data data(model);
    ConstraintSolver solver(model, {std::make_shared<Distance>(Marker{0, Transform3()},
                                                               Marker{1, Transform3()}, L)});
    auto rod_force = [&](const Vec3& p, const Vec3& velocity) {
        VecX v = VecX::Zero(6);
        v.tail<3>() = velocity;   // body axes are world axes here
        solver.forward_dynamics(data, free_configuration(p, Quat::Identity()), v, VecX::Zero(6), 0.0);
        return Vec6(solver.J().transpose() * solver.lambda());
    };

    // Horizontal, at rest: gravity is across the rod, which pulls nothing.
    CHECK(rod_force(Vec3(L, 0.0, 0.0), Vec3::Zero()).norm() < 1e-12);

    // Hanging, at rest: the rod carries the weight.
    const Vec6 hanging = rod_force(Vec3(0.0, 0.0, -L), Vec3::Zero());
    CHECK((hanging.tail<3>() - Vec3(0.0, 0.0, m * g_accel)).norm() < 1e-12);
    CHECK(hanging.head<3>().norm() < 1e-12);

    // At the bottom, moving at 2 m/s: weight plus centripetal force.
    const Vec6 swinging = rod_force(Vec3(0.0, 0.0, -L), Vec3(2.0, 0.0, 0.0));
    CHECK((swinging.tail<3>() - Vec3(0.0, 0.0, m * g_accel + m * 4.0 / L)).norm() < 1e-12);
}

TEST_CASE("Kernel analytic: a large swing takes the period of the elliptic integral",
          "[kernel][analytic]")
{
    // From rest at theta0 = 1 rad, a pendulum of length L swings with period
    // T = 4 sqrt(L/g) K(sin(theta0/2)). Between its two passes through the
    // bottom half a period goes by.
    const Real L = 1.0, theta0 = 1.0;
    const Model model = free_body(RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)),
                                  Vec3(0.0, 0.0, -g_accel));
    Data data(model);
    ConstraintSolver solver(model, {std::make_shared<Distance>(Marker{0, Transform3()},
                                                               Marker{1, Transform3()}, L)});
    VecX q = free_configuration(Vec3(L * std::sin(theta0), 0.0, -L * std::cos(theta0)), Quat::Identity());
    VecX v = VecX::Zero(6);

    const Real dt = 1e-3;
    std::vector<Real> bottom_passes;
    Real t = 0.0, x_prev = q(0);
    while (bottom_passes.size() < 2 && t < 10.0) {
        rk4_constrained(model, solver, data, q, v, t, dt);
        t += dt;
        const Real x = q(0);
        if ((x_prev > 0.0) != (x > 0.0)) {
            bottom_passes.push_back(t - dt * x / (x - x_prev));   // linear interpolation
        }
        x_prev = x;
    }
    REQUIRE(bottom_passes.size() == 2);
    const Real half_period = 2.0 * std::sqrt(L / g_accel) * elliptic_K(std::sin(0.5 * theta0));
    CHECK(std::abs((bottom_passes[1] - bottom_passes[0]) - half_period) < 1e-6);
}

TEST_CASE("Kernel analytic: a bar released from a hinge loads it with a quarter of its weight",
          "[kernel][analytic]")
{
    // A uniform bar, hinged at one end about the world Y axis, released
    // horizontally from rest. Its angular acceleration is
    // alpha = m g (L/2) / I_hinge and the hinge carries
    // m g - m alpha (L/2) = m g (1 - m (L/2)^2 / I_hinge),
    // which is m g / 4 for a thin rod.
    const Real m = 2.0, L = 1.0;
    RigidBodyInertia bar = RigidBodyInertia::from_solid_box(m, Vec3(0.5 * L, 0.02, 0.02));
    const Model model = free_body(bar, Vec3(0.0, 0.0, -g_accel));
    Data data(model);

    // Hinge axis along world Y: marker Z turned onto Y, at the origin (ground)
    // and at the bar's end (bar frame = world at the start).
    const Quat z_to_y(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX()));
    ConstraintSolver solver(model, {revolute_closure(Marker{0, Transform3(z_to_y, Vec3::Zero())},
                                                     Marker{1, Transform3(z_to_y, Vec3(-0.5 * L, 0, 0))})});
    const VecX q = free_configuration(Vec3(0.5 * L, 0.0, 0.0), Quat::Identity());
    const VecX a = solver.forward_dynamics(data, q, VecX::Zero(6), VecX::Zero(6), 0.0);
    CHECK(solver.phi().norm() < 1e-14);

    const Real I_hinge = bar.I_com_B(1, 1) + m * 0.25 * L * L;
    const Real alpha = m * g_accel * 0.5 * L / I_hinge;
    CHECK((a.head<3>() - Vec3(0.0, alpha, 0.0)).norm() < 1e-11);   // falls turning about +Y

    const Vec6 hinge = solver.J().transpose() * solver.lambda();
    const Real lift = m * g_accel * (1.0 - m * 0.25 * L * L / I_hinge);
    CHECK((hinge.tail<3>() - Vec3(0.0, 0.0, lift)).norm() < 1e-11);
    CHECK(lift > 0.24 * m * g_accel);   // close to a quarter: the bar is thin
    CHECK(lift < 0.26 * m * g_accel);
}
