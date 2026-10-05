// Closed-form physics of each joint, on the kernel (plan task 2.7b). These
// replace the analytic tests of the legacy joints and algorithms
// (test_revolute_joint, test_prismatic_joint, test_other_joints,
// test_algorithms, test_multibody_chains, test_loop_constraints).
//
// Gravity is -Z. A pendulum's bob hangs a distance L below its pivot, and
// I_p is its inertia about the pivot axis: the compound pendulum swings at
// omega^2 = m g L / I_p for small angles.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;

namespace {

constexpr Real m_bob = 2.0, L_bob = 0.9;

/// A bob of mass m_bob, a small box whose centre of mass is L_bob below the
/// body origin (the pivot).
RigidBodyInertia bob()
{
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(m_bob, Vec3(0.05, 0.04, 0.03));
    I.com_B = Vec3(0.0, 0.0, -L_bob);
    return I;
}

/// Frames whose Z axis is the world's Y.
Transform3 z_along_y()
{
    return Transform3(Quat(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX())), Vec3::Zero());
}

/// Time between the first two upward zero crossings of x(t), linearly
/// interpolated: one period.
Real period_of(const std::vector<Real>& t, const std::vector<Real>& x)
{
    std::vector<Real> up;
    for (std::size_t k = 1; k < x.size() && up.size() < 2; ++k) {
        if (x[k - 1] < 0.0 && x[k] >= 0.0) {
            up.push_back(t[k - 1] + (t[k] - t[k - 1]) * (-x[k - 1]) / (x[k] - x[k - 1]));
        }
    }
    REQUIRE(up.size() == 2);
    return up[1] - up[0];
}

/// Period of coordinate k of a simulator, released from its current state.
Real simulated_period(Simulator& sim, int k, Real duration, Real dt)
{
    std::vector<Real> t{sim.time}, x{sim.q(k)};
    const int steps = static_cast<int>(std::round(duration / dt));
    for (int s = 0; s < steps; ++s) {
        sim.step(dt);
        t.push_back(sim.time);
        x.push_back(sim.q(k));
    }
    return period_of(t, x);
}

Real energy(const Model& model, const Data& data)
{
    return kinetic_energy(model, data) + potential_energy(model, data);
}

} // namespace

// --- Revolute ------------------------------------------------------------------

TEST_CASE("Kernel physics: revolute pendulum swings at sqrt(m g L / I_p)", "[kernel][physics]")
{
    // A small swing (0.01 rad) lengthens the period by theta0^2 / 16 = 6e-6.
    System sys;
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_along_y(), z_along_y(), bob());
    Simulator sim(sys);
    sim.q(0) = 0.01;
    sim.initialize();

    const Real I_p = bob().I_com_B(1, 1) + m_bob * L_bob * L_bob;
    const Real T = 2.0 * pi * std::sqrt(I_p / (m_bob * g_accel * L_bob));
    CHECK(std::abs(simulated_period(sim, 0, 2.5 * T, 1e-3) - T) < 1e-4 * T);
}

TEST_CASE("Kernel physics: revolute pendulum statics, torque and energy", "[kernel][physics]")
{
    System sys;
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_along_y(), z_along_y(), bob());
    Data data(sys.model);
    const Real I_p = bob().I_com_B(1, 1) + m_bob * L_bob * L_bob;

    // Hanging straight down at rest: no acceleration.
    VecX q = VecX::Zero(1), v = VecX::Zero(1), tau = VecX::Zero(1);
    CHECK(std::abs(aba(sys.model, data, q, v, tau)(0)) < 1e-12);

    // A torque about the axis, hanging: alpha = tau / I_p.
    tau(0) = 3.0;
    CHECK(std::abs(aba(sys.model, data, q, v, tau)(0) - 3.0 / I_p) < 1e-12);

    // Horizontal (q = pi/2): gravity's torque m g L, falling back.
    q(0) = 0.5 * pi;
    CHECK(std::abs(rnea(sys.model, data, q, v, VecX::Zero(1))(0) - m_bob * g_accel * L_bob) < 1e-12);

    // Released from 1 rad, the energy stays put over 5 s of RK4 at 1 ms.
    Simulator sim(sys);
    sim.q(0) = 1.0;
    sim.initialize();
    const Real E0 = energy(sys.model, sim.data());
    sim.run(5.0, 1e-3);
    CHECK(std::abs(energy(sys.model, sim.data()) - E0) < 1e-9 * std::abs(E0) + 1e-9);
}

// --- Prismatic -----------------------------------------------------------------

TEST_CASE("Kernel physics: sliders under gravity", "[kernel][physics]")
{
    // Vertical: falls at g. Horizontal: stays. At 45 degrees: g / sqrt(2).
    auto slide_acceleration = [](const Vec3& axis) {
        Model model;
        const Vec3 z = axis.normalized();
        const Quat R = Quat::FromTwoVectors(Vec3::UnitZ(), z);
        model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(R, Vec3::Zero()),
                       Transform3(R, Vec3::Zero()),
                       RigidBodyInertia::from_solid_box(3.0, Vec3(0.1, 0.1, 0.1)));
        Data data(model);
        return aba(model, data, VecX::Zero(1), VecX::Zero(1), VecX::Zero(1))(0);
    };
    CHECK(std::abs(slide_acceleration(Vec3::UnitZ()) + g_accel) < 1e-12);
    CHECK(std::abs(slide_acceleration(Vec3::UnitX())) < 1e-12);
    CHECK(std::abs(slide_acceleration(Vec3(1.0, 0.0, 1.0)) + g_accel / std::sqrt(2.0)) < 1e-12);
}

TEST_CASE("Kernel physics: a slider on a spring oscillates at sqrt(k / m)", "[kernel][physics]")
{
    // Horizontal slider of 3 kg on a 400 N/m spring anchored at the origin,
    // rest length 0.5 m: omega = sqrt(400 / 3). A force of 6 N gives a = 2.
    const Real m = 3.0, k = 400.0;
    System sys;
    const Quat R = Quat::FromTwoVectors(Vec3::UnitZ(), Vec3::UnitX());
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(R, Vec3::Zero()),
                       Transform3(R, Vec3::Zero()),
                       RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.1)));
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, 1, Vec3::Zero(), Vec3::Zero(), k, 0.0, 0.5));

    Simulator sim(sys);
    sim.q(0) = 0.55;   // 5 cm stretch; the spring's length is the coordinate
    sim.initialize();
    const Real T = 2.0 * pi * std::sqrt(m / k);
    // Oscillation about 0.5: the period of q - 0.5.
    std::vector<Real> t{sim.time}, x{sim.q(0) - 0.5};
    for (int s = 0; s < static_cast<int>(2.5 * T / 1e-4); ++s) {
        sim.step(1e-4);
        t.push_back(sim.time);
        x.push_back(sim.q(0) - 0.5);
    }
    CHECK(std::abs(period_of(t, x) - T) < 1e-6 * T);

    Data data(sys.model);
    CHECK(std::abs(aba(sys.model, data, VecX::Constant(1, 0.5), VecX::Zero(1), VecX::Constant(1, 6.0))(0)
                   - 6.0 / m) < 1e-12);
}

// --- Spherical, universal, fixed --------------------------------------------------

TEST_CASE("Kernel physics: spherical pendulum swings like the compound pendulum",
          "[kernel][physics]")
{
    // Released in the XZ plane, it swings about Y with the same period as a
    // hinge about Y.
    System sys;
    sys.model.add_body(0, std::make_shared<SphericalJointModel>(), Transform3(), Transform3(), bob());
    Simulator sim(sys);
    sim.q.head<4>() = Quat(Eigen::AngleAxisd(0.01, Vec3::UnitY())).coeffs();
    sim.initialize();

    const Real I_p = bob().I_com_B(1, 1) + m_bob * L_bob * L_bob;
    const Real T = 2.0 * pi * std::sqrt(I_p / (m_bob * g_accel * L_bob));
    // The quaternion's y component is sin(theta / 2): it crosses zero with theta.
    CHECK(std::abs(simulated_period(sim, 1, 2.5 * T, 1e-3) - T) < 1e-4 * T);
}

TEST_CASE("Kernel physics: a symmetric body spins steadily on a spherical joint",
          "[kernel][physics]")
{
    // Pivot at the centre of mass, no gravity, spinning about the symmetry
    // axis: the angular velocity stays constant.
    System sys;
    sys.model.gravity.setZero();
    sys.model.add_body(0, std::make_shared<SphericalJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(2.0, Vec3(0.2, 0.2, 0.05)));
    Simulator sim(sys);
    sim.v << 0.0, 0.0, 7.0;
    sim.initialize();
    sim.run(2.0, 1e-3);
    CHECK((sim.v - Vec3(0.0, 0.0, 7.0)).norm() < 1e-12);
    // It has turned by 14 rad about Z.
    CHECK(std::abs(std::remainder(2.0 * std::atan2(sim.q(2), sim.q(3)) - 14.0, 2.0 * pi)) < 1e-9);
}

TEST_CASE("Kernel physics: universal joint pendulum and torque", "[kernel][physics]")
{
    // First axis vertical (Z), second the rotated X: the bob swings about X.
    System sys;
    sys.model.add_body(0, std::make_shared<UniversalJointModel>(), Transform3(), Transform3(), bob());
    const Real I_p = bob().I_com_B(0, 0) + m_bob * L_bob * L_bob;
    const Real T = 2.0 * pi * std::sqrt(I_p / (m_bob * g_accel * L_bob));

    Simulator sim(sys);
    sim.q(1) = 0.01;
    sim.initialize();
    CHECK(std::abs(simulated_period(sim, 1, 2.5 * T, 1e-3) - T) < 1e-4 * T);

    // A torque about the vertical axis turns the hanging bob at tau / I_zz.
    Data data(sys.model);
    VecX tau = VecX::Zero(2);
    tau(0) = 2.0;
    const VecX a = aba(sys.model, data, VecX::Zero(2), VecX::Zero(2), tau);
    CHECK(std::abs(a(0) - 2.0 / bob().I_com_B(2, 2)) < 1e-12);
    CHECK(std::abs(a(1)) < 1e-12);
}

TEST_CASE("Kernel physics: a fixed joint holds its body at the joint frames", "[kernel][physics]")
{
    Model model;
    const Transform3 X_PJ(Quat(Eigen::AngleAxisd(0.3, Vec3::UnitY())), Vec3(0.2, -0.1, 0.4));
    const Transform3 X_CJ(Quat(Eigen::AngleAxisd(-0.2, Vec3::UnitX())), Vec3(0.0, 0.1, 0.0));
    model.add_body(0, std::make_shared<FixedJointModel>(), X_PJ, X_CJ, bob());
    // Something moving below it, so the model has dynamics.
    model.add_body(1, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), bob());
    REQUIRE(model.nv == 1);

    Data data(model);
    forward_kinematics(model, data, VecX::Zero(1));
    const Transform3 expected = X_PJ * X_CJ.inverse();
    CHECK((data.oMi[1].p - expected.p).norm() < 1e-15);
    CHECK(data.oMi[1].q.angularDistance(expected.q) < 1e-15);
}

// --- Chains --------------------------------------------------------------------

TEST_CASE("Kernel physics: double pendulum slow mode at (2 - sqrt 2) g / L", "[kernel][physics]")
{
    // Two equal point masses on equal massless rods: the normal modes are
    // omega^2 = (2 -+ sqrt 2) g / L, the slow one with the lower rod's
    // angle sqrt 2 times the upper one's. The bobs here are tiny boxes, which
    // changes the frequency by about 1e-4 relative.
    const Real m = 1.0, L = 1.0;
    RigidBodyInertia point = RigidBodyInertia::from_solid_box(m, Vec3(0.005, 0.005, 0.005));
    point.com_B = Vec3(0.0, 0.0, -L);
    System sys;
    const int upper = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_along_y(),
                                         z_along_y(), point);
    sys.model.add_body(upper, std::make_shared<RevoluteJointModel>(),
                       Transform3(z_along_y().q, Vec3(0.0, 0.0, -L)), z_along_y(), point);

    // Relative coordinates: theta2 = sqrt 2 theta1 absolute, so q1 = theta1
    // and q2 = (sqrt 2 - 1) theta1.
    Simulator sim(sys);
    sim.q(0) = 0.01;
    sim.q(1) = (std::sqrt(2.0) - 1.0) * 0.01;
    sim.initialize();

    const Real omega = std::sqrt((2.0 - std::sqrt(2.0)) * g_accel / L);
    const Real T = 2.0 * pi / omega;
    CHECK(std::abs(simulated_period(sim, 0, 2.5 * T, 1e-3) - T) < 1e-3 * T);
}

// --- Constraints -----------------------------------------------------------------

TEST_CASE("Kernel physics: a coincident point holds two bodies together under load",
          "[kernel][physics]")
{
    // Two free bodies joined at a point, pulled apart by equal and opposite
    // forces: they stay joined and the joint carries the pull.
    System sys;
    sys.model.gravity.setZero();
    const auto box = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    const int a = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(),
                                     Transform3(), box);
    const int b = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(),
                                     Transform3(), box);
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{a, Transform3::FromTranslation(Vec3(0.2, 0.0, 0.0))},
        Marker{b, Transform3::FromTranslation(Vec3(-0.2, 0.0, 0.0))}));

    Simulator sim(sys);
    sim.q.segment<3>(sys.model.idx_q[b]) = Vec3(0.4, 0.0, 0.0);   // joined at x = 0.2
    sim.initialize();
    // 50 N on each, apart along X, through their centres of mass.
    sim.force_callback = [&](Simulator&, Real, VecX& tau) {
        tau(sys.model.idx_v[a] + 3) -= 50.0;
        tau(sys.model.idx_v[b] + 3) += 50.0;
    };
    sim.run(1.0, 1e-3);

    Data data(sys.model);
    sim.solver().evaluate(data, sim.q, sim.v, sim.time);
    CHECK(sim.solver().phi().norm() < 1e-10);
    CHECK(sim.v.norm() < 1e-9);   // balanced: nothing moves

    // The joint's force on b is the 50 N pull, back toward a.
    sim.acceleration(sim.q, sim.v, sim.time);
    const VecX f = sim.solver().J().transpose() * sim.solver().lambda();
    CHECK(std::abs(f(sys.model.idx_v[b] + 3) + 50.0) < 1e-9);
}

TEST_CASE("Kernel physics: a height driver holds a body point while the body swings",
          "[kernel][physics]")
{
    // A free body under gravity with one of its points held at z = 0.3 by a
    // dot-2 driver: that point does not fall, whatever the body does.
    System sys;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.2, 0.1, 0.1)));
    const Vec3 point(0.2, 0.0, 0.0);
    sys.constraints.push_back(std::make_shared<Dot2>(
        Marker{0, Transform3()}, 2, Marker{1, Transform3::FromTranslation(point)},
        TimeFunction::constant(0.3)));

    Simulator sim(sys);
    sim.q(2) = 0.3;                // body origin at the target height...
    sim.v << 0.5, -0.3, 1.0, 0.2, 0.0, 0.0;
    sim.initialize();              // ...and velocities made consistent

    Real worst_height_error = 0.0, lowest_centre = 1.0;
    for (int step = 0; step < 1000; ++step) {
        sim.step(1e-3);
        const RigidBodyState& s = sim.states()[1];
        worst_height_error = std::max(worst_height_error, std::abs(s.pose_WB().apply(point).z() - 0.3));
        lowest_centre = std::min(lowest_centre, s.p_WB.z());
    }
    CHECK(worst_height_error < 1e-9);
    CHECK(sim.projection_failures() == 0);
    // Gravity still acts: the centre of mass, 0.2 m from the held point,
    // swings down below it.
    CHECK(lowest_centre < 0.25);
}
