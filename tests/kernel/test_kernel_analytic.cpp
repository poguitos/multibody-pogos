// Closed-form cases on the kernel (plan task 2.7). They replace the tests of
// the single-body integrator (dynamics_og.hpp) and of the body-level
// constraint solver (solvers/solver.hpp), both deleted:
//
//   - a free body moves at constant velocity, falls freely, and turns as
//     Euler's equations say, including the torque-free symmetric top;
//   - a torque-free body with three unequal moments of inertia turns as
//     Jacobi's elliptic functions say, and with their period (task 5.1);
//   - a heavy symmetric top on a spherical joint precesses steadily at the
//     rate of the closed-form condition, and, released with spin only, nods
//     down to the root of its energy cubic (task 5.1);
//   - a pendulum held by a distance constraint pulls with m g and
//     m g + m v^2 / L, and swings with the period of the elliptic integral;
//   - a bar released horizontally from a hinge loads it with
//     m g (1 - m (L/2)^2 / I_hinge), a quarter of its weight for a thin rod.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/spatial/spatial.hpp"

#include "kernel/kernel_helpers.hpp"
#include "support/elliptic.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::elliptic_K;
using mbd_test::jacobi;

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

// --- Plan task 5.1: more closed forms ---------------------------------------------

TEST_CASE("Kernel analytic: a torque-free body with unequal inertias turns as Jacobi's functions say",
          "[kernel][analytic]")
{
    // Principal moments I1 < I2 < I3, no torque, started at w = (w1, 0, w3)
    // with L^2 > 2 E I2 (turning about the axis of the largest moment). Then
    // (Landau and Lifshitz, Mechanics, section 37)
    //
    //   w1 = sqrt((2 E I3 - L^2) / (I1 (I3 - I1))) cn(tau, k)
    //   w2 = sqrt((2 E I3 - L^2) / (I2 (I3 - I2))) sn(tau, k)
    //   w3 = sqrt((L^2 - 2 E I1) / (I3 (I3 - I1))) dn(tau, k)
    //
    // with tau = t sqrt((I3 - I2)(L^2 - 2 E I1) / (I1 I2 I3)) and
    // k^2 = (I2 - I1)(2 E I3 - L^2) / ((I3 - I2)(L^2 - 2 E I1)); w2 is positive
    // at first because I2 w2' = (I3 - I1) w3 w1 > 0. The angular velocity
    // repeats with the period 4 K(k) / (dtau/dt). In the world, the angular
    // momentum R I w stays fixed.
    //
    // RK4 at 1 ms: the rates are below 3.2 rad/s, so its local error is about
    // (w h)^5 / 120 = 3e-15 of the state per step, 1e-11 over the 5000 steps
    // to 5 s (and over one period); allowed 1e-9.
    const RigidBodyInertia body = RigidBodyInertia::from_solid_box(3.0, Vec3(0.3, 0.2, 0.1));
    const Model model = free_body(body, Vec3::Zero());
    Data data(model);
    const Real I1 = body.I_com_B(0, 0), I2 = body.I_com_B(1, 1), I3 = body.I_com_B(2, 2);
    REQUIRE(I1 < I2);
    REQUIRE(I2 < I3);
    const Real w10 = 1.0, w30 = 3.0;
    const Real E2 = I1 * w10 * w10 + I3 * w30 * w30;            // 2 E
    const Real L2 = I1 * I1 * w10 * w10 + I3 * I3 * w30 * w30;  // L^2
    REQUIRE(L2 > E2 * I2);
    const Real A1 = std::sqrt((E2 * I3 - L2) / (I1 * (I3 - I1)));
    const Real A2 = std::sqrt((E2 * I3 - L2) / (I2 * (I3 - I2)));
    const Real A3 = std::sqrt((L2 - E2 * I1) / (I3 * (I3 - I1)));
    const Real rate = std::sqrt((I3 - I2) * (L2 - E2 * I1) / (I1 * I2 * I3));
    const Real m = (I2 - I1) * (E2 * I3 - L2) / ((I3 - I2) * (L2 - E2 * I1));
    auto closed = [&](Real t) {
        Real sn = 0.0, cn = 0.0, dn = 0.0;
        jacobi(rate * t, m, sn, cn, dn);
        return Vec3(A1 * cn, A2 * sn, A3 * dn);
    };
    CHECK((closed(0.0) - Vec3(w10, 0.0, w30)).norm() < 1e-14);   // the amplitudes fit the start

    const Quat R0(Eigen::AngleAxisd(0.7, Vec3(0.3, -1.0, 0.4).normalized()));
    VecX q = free_configuration(Vec3::Zero(), R0);
    VecX v = VecX::Zero(6);
    v.head<3>() = Vec3(w10, 0.0, w30);
    const Mat3 I = body.I_com_B;
    const Vec3 L0 = R0 * (I * v.head<3>());

    const Real dt = 1e-3;
    const Real period = 4.0 * elliptic_K(std::sqrt(m)) / rate;
    Real worst = 0.0, worst_L = 0.0;
    for (int step = 1; step <= 5000; ++step) {
        mbd_test::rk4_step(model, data, q, v, dt);
        if (step % 250 == 0) {
            worst = std::max(worst, (v.head<3>() - closed(step * dt)).norm());
            const Quat R(q.segment<4>(3));
            worst_L = std::max(worst_L, (R * (I * v.head<3>()) - L0).norm());
        }
    }
    INFO("k^2 = " << m << ", period " << period << " s; largest error of w " << worst << " rad/s, of the world "
                  << "angular momentum " << worst_L << " N m s (of " << L0.norm() << ")");
    CHECK(m > 0.05);   // far from the symmetric top (k = 0): the motion is elliptic
    CHECK(worst < 1e-9);
    CHECK(worst_L < 1e-9 * L0.norm());

    // After one period the angular velocity is back at its start.
    VecX q1 = free_configuration(Vec3::Zero(), R0);
    VecX v1 = VecX::Zero(6);
    v1.head<3>() = Vec3(w10, 0.0, w30);
    const int n = static_cast<int>(std::floor(period / dt));
    for (int step = 0; step < n; ++step) mbd_test::rk4_step(model, data, q1, v1, dt);
    mbd_test::rk4_step(model, data, q1, v1, period - n * dt);
    INFO("after one period: " << v1.head<3>().transpose());
    CHECK((v1.head<3>() - Vec3(w10, 0.0, w30)).norm() < 1e-9);
}

namespace {

/// A symmetric top on a spherical joint at its tip, the ground's origin:
/// mass m, its centre of mass l above the tip along the body's Z axis,
/// moments A about the tip across the axis and C along it.
struct HeavyTop {
    Real m{1.0}, l{0.05};
    RigidBodyInertia inertia;
    Model model;
    Real A{0.0}, C{0.0};

    HeavyTop()
    {
        inertia = RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.02));
        inertia.com_B = Vec3(0.0, 0.0, l);
        model.gravity = Vec3(0.0, 0.0, -g_accel);
        model.add_body(0, std::make_shared<SphericalJointModel>(), Transform3(), Transform3(), inertia);
        A = inertia.I_com_B(0, 0) + m * l * l;
        C = inertia.I_com_B(2, 2);
    }

    /// The axis, tilted by theta from the vertical about world X.
    static Quat tilted(Real theta) { return Quat(Eigen::AngleAxisd(theta, Vec3::UnitX())); }
};

/// The top's axis in the world.
Vec3 top_axis(const VecX& q)
{
    return Quat(q.head<4>()) * Vec3::UnitZ();
}

} // namespace

TEST_CASE("Kernel analytic: a heavy top precesses steadily at the closed-form rate", "[kernel][analytic]")
{
    // A top tilted by theta from the vertical, spinning with w3 about its
    // axis, precesses steadily about the vertical at Omega when (Goldstein,
    // Classical Mechanics, 5.7)
    //
    //   m g l = Omega (C w3 - A Omega cos theta),
    //
    // a quadratic whose smaller root is the slow precession. Started there,
    // with body angular velocity (0, Omega sin theta, w3) (the precession
    // Omega about world Z seen in the tilted body, plus the spin), the tilt
    // must stay at theta and the axis turn about Z by Omega t.
    //
    // RK4 at 0.1 ms: the fastest rate is 50 rad/s, so its local error is about
    // (w h)^5 / 120 = 3e-14 of the state per step, 5e-10 over the 15000 steps
    // to 1.5 s; allowed 1e-8 rad on the tilt and on the precession angle.
    HeavyTop top;
    Data data(top.model);
    const Real theta = 0.5, w3 = 50.0;
    const Real a = top.A * std::cos(theta), b = top.C * w3, c = top.m * g_accel * top.l;
    const Real Omega = (b - std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);

    VecX q = top.tilted(theta).coeffs();
    VecX v(3);
    v << 0.0, Omega * std::sin(theta), w3;
    const Vec3 axis0 = top_axis(q);
    const Real dt = 1e-4, T = 1.5;
    Real worst_tilt = 0.0;
    for (int step = 0; step < 15000; ++step) {
        mbd_test::rk4_step(top.model, data, q, v, dt);
        worst_tilt = std::max(worst_tilt, std::abs(std::acos(top_axis(q).z()) - theta));
    }
    const Vec3 axis = top_axis(q);
    const Vec2 h0(axis0.x(), axis0.y()), h(axis.x(), axis.y());
    const Real turned = std::atan2(h0.x() * h.y() - h0.y() * h.x(), h0.dot(h));
    INFO("Omega " << Omega << " rad/s; turned " << turned << " rad against " << Omega * T << " (by "
                  << turned - Omega * T << "); largest change of tilt " << worst_tilt << " rad");
    CHECK(Omega > 1.0);   // a real precession, well within half a turn over T
    CHECK(Omega * T < 3.0);
    CHECK(worst_tilt < 1e-8);
    CHECK(std::abs(turned - Omega * T) < 1e-8);
    CHECK(std::abs(v(2) - w3) < 1e-10);   // a symmetric top keeps its spin
}

TEST_CASE("Kernel analytic: a heavy top released with spin only nods to the root of its energy cubic",
          "[kernel][analytic]")
{
    // Released at tilt theta0 with spin w3 and no other motion, the top's
    // u = cos theta obeys (Goldstein, 5.7)
    //
    //   (du/dt)^2 = (u0 - u) [beta (1 - u^2) - a^2 (u0 - u)],
    //   a = C w3 / A,  beta = 2 m g l / A,
    //
    // so it nods between u0 and the root u1 < u0 of
    // beta u^2 - a^2 u + (a^2 u0 - beta) = 0: the smaller root, the larger
    // lying above 1. The lowest u sampled at every step, refined by a
    // parabola through it and its neighbours, must reach u1. The parabola's
    // error is of order dt^3 times the third derivative of u, about
    // (u0 - u1) a^3: 0.013 x 56^3 x 1e-12 = 2e-9 at 0.1 ms. RK4's error is
    // below 1e-9 (as for the steady top). Allowed 1e-7, under 1e-5 of the nod
    // (u0 - u1 = 0.013).
    HeavyTop top;
    Data data(top.model);
    const Real theta0 = 0.5, w3 = 50.0;
    const Real u0 = std::cos(theta0);
    const Real a = top.C * w3 / top.A, beta = 2.0 * top.m * g_accel * top.l / top.A;
    const Real root = std::sqrt(a * a * a * a - 4.0 * beta * (a * a * u0 - beta));
    const Real u1 = (a * a - root) / (2.0 * beta);
    const Real other = (a * a + root) / (2.0 * beta);
    REQUIRE(u1 < u0);
    REQUIRE(other > 1.0);   // the root beyond the physical range: u1 is the one reached

    VecX q = top.tilted(theta0).coeffs();
    VecX v(3);
    v << 0.0, 0.0, w3;
    const Real dt = 1e-4;
    Real u_prev2 = u0, u_prev = u0, lowest = u0;
    int dips = 0;
    for (int step = 0; step < 3000; ++step) {
        mbd_test::rk4_step(top.model, data, q, v, dt);
        const Real u = top_axis(q).z();
        if (step >= 1 && u_prev < u_prev2 && u_prev <= u) {   // a sampled minimum
            const Real curvature = u - 2.0 * u_prev + u_prev2;
            const Real refined = u_prev - (u - u_prev2) * (u - u_prev2) / (8.0 * curvature);
            lowest = std::min(lowest, refined);
            ++dips;
            INFO("nod " << dips << ": lowest u " << refined << ", closed form " << u1 << " (by " << refined - u1
                        << ")");
            CHECK(std::abs(refined - u1) < 1e-7);
        }
        u_prev2 = u_prev;
        u_prev = u;
    }
    INFO("u0 " << u0 << ", u1 " << u1 << " (nod " << u0 - u1 << "); lowest reached " << lowest << " in " << dips
               << " nods");
    CHECK(dips >= 2);
    CHECK(std::abs(lowest - u1) < 1e-7);
}
