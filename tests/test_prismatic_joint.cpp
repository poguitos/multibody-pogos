#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/system.hpp"
#include "mbd/simulator.hpp"
#include "mbd/joint.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    // Slider fixture: a 1 kg solid cube body connected to ground by a
    // prismatic joint along the joint frame's Z axis.
    //
    // The joint orientation is configurable via X_PJ:
    //   - default: joint Z = world Z (slider moves in world Z)
    //   - rotated: joint Z = some other world direction
    struct SliderFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
        mbd::Real mass{1.0};
    };

    /// Build a slider with the joint Z axis rotated such that joint Z = `axis_W`.
    /// `axis_W` must be a unit vector.
    SliderFixture make_slider(const mbd::Vec3& axis_W = mbd::Vec3::UnitZ(),
                              mbd::Real mass = 1.0)
    {
        using namespace mbd;
        SliderFixture fx;
        fx.mass = mass;

        // Build a rotation that maps joint Z to axis_W
        // We use Eigen::Quaterniond::FromTwoVectors
        Quat q_PJ = Quat::FromTwoVectors(Vec3::UnitZ(), axis_W);
        Transform3 X_PJ(q_PJ, Vec3::Zero());

        auto inertia = RigidBodyInertia::from_solid_box(mass, Vec3(0.5, 0.5, 0.5));
        fx.body_idx = fx.sys.add_body(inertia, RigidBodyState{}, "slider", kGroundIndex);
        fx.sys.add_joint(std::make_unique<PrismaticCoordJoint>(
            X_PJ, Transform3::Identity(),
            kGroundIndex, fx.body_idx));

        return fx;
    }
}

// ============================================================================
// Kinematics
// ============================================================================

TEST_CASE("PrismaticJoint: q=0 places body at parent origin",
          "[prismatic][kinematics]")
{
    using namespace mbd;
    auto fx = make_slider();

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("PrismaticJoint: q translates body along world Z (default axis)",
          "[prismatic][kinematics]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitZ()); // joint Z = world Z

    fx.sys.q.setZero();
    fx.sys.q(0) = 1.5;
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(1.5, 1e-9));
}

TEST_CASE("PrismaticJoint: q translates body along world X when axis is X",
          "[prismatic][kinematics]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitX()); // joint Z = world X

    fx.sys.q.setZero();
    fx.sys.q(0) = 2.0;
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("PrismaticJoint: q translates body along world Y when axis is Y",
          "[prismatic][kinematics]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitY()); // joint Z = world Y

    fx.sys.q.setZero();
    fx.sys.q(0) = 0.7;
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.7, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("PrismaticJoint: q_dot produces velocity along joint axis",
          "[prismatic][kinematics]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitX());

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 3.0;
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("PrismaticJoint: angular velocity is zero (pure translation)",
          "[prismatic][kinematics]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitX());

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 5.0;
    fx.sys.compute_kinematics();

    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    REQUIRE_THAT(w.norm(), WithinAbs(0.0, 1e-12));
}

// ============================================================================
// Force-acceleration relationship
// ============================================================================

TEST_CASE("PrismaticJoint: tau gives a = F/m along axis",
          "[prismatic][force]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitX(), 2.0); // 2 kg slider along X

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real F = 8.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    // Newton: a = F/m = 4, after 1s: v = 4, p = 2
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(4.0, 0.01));
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(2.0, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(2.0, 0.01));
}

TEST_CASE("PrismaticJoint: rotated axis - tau gives acceleration along that axis",
          "[prismatic][force]")
{
    using namespace mbd;
    // Slider with joint Z aligned to world (1,1,0)/sqrt(2)
    Vec3 axis_W(1.0, 1.0, 0.0);
    axis_W.normalize();
    auto fx = make_slider(axis_W);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real F = 1.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    // a along axis = F/m = 1, after 1s: q = 0.5
    // World position = q * axis_W = 0.5 * (1/sqrt(2), 1/sqrt(2), 0)
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.5, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(),
                 WithinAbs(0.5 / std::sqrt(2.0), 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(),
                 WithinAbs(0.5 / std::sqrt(2.0), 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-6));
}

// ============================================================================
// Gravity tests
// ============================================================================

TEST_CASE("PrismaticJoint: vertical slider falls under gravity at g",
          "[prismatic][gravity]")
{
    using namespace mbd;
    // Slider with joint Z = world Y (so q increases upward).
    auto fx = make_slider(Vec3::UnitY());

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q(0) = 5.0; // start 5m above ground
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(1.0, 0.001);

    // After 1s of free fall: v = -g (downward), p = 5 - 0.5*g
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(-g_accel, 0.01));
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(5.0 - 0.5 * g_accel, 0.01));
}

TEST_CASE("PrismaticJoint: horizontal slider does not move under gravity",
          "[prismatic][gravity]")
{
    using namespace mbd;
    // Slider with joint Z = world X (perpendicular to gravity)
    auto fx = make_slider(Vec3::UnitX());

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    // Gravity perpendicular to slider axis - slider should not move.
    // (The joint constraint reacts the gravity component perpendicular to
    // the axis; only the parallel component would move it.)
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(0.0, 1e-6));
}

TEST_CASE("PrismaticJoint: 45-degree inclined slider accelerates at g/sqrt(2)",
          "[prismatic][gravity]")
{
    using namespace mbd;
    // Inclined slider: joint Z = (cos(pi/4), sin(pi/4), 0) — lies in XY plane,
    // 45° above horizontal. Gravity component along axis = -g*sin(45°) = -g/sqrt(2).
    Vec3 axis_W(std::cos(pi / 4.0), std::sin(pi / 4.0), 0.0);
    auto fx = make_slider(axis_W);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q(0) = 5.0;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(1.0, 0.001);

    // Acceleration along axis: -g * sin(45°) = -g/sqrt(2)
    // After 1s: v = -g/sqrt(2), q = 5 - 0.5*g/sqrt(2)
    const Real a_expected = -g_accel / std::sqrt(2.0);

    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(a_expected, 0.05));
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(5.0 + 0.5 * a_expected, 0.05));
}

// ============================================================================
// Spring-mass oscillator
// ============================================================================

TEST_CASE("PrismaticJoint: spring-mass oscillator has period T = 2*pi*sqrt(m/k)",
          "[prismatic][spring][energy]")
{
    using namespace mbd;
    // Horizontal slider (no gravity effect along axis), with linear spring
    // pulling toward q = 0.
    const Real m = 1.0;
    const Real k = 100.0; // N/m

    auto fx = make_slider(Vec3::UnitX(), m);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero()); // remove gravity entirely for clean test
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start displaced by 0.1, no velocity
    fx.sys.q.setZero();
    fx.sys.q(0) = 0.1;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
        tau(0) += -k * s.q(0);
    };

    const Real T_expected = 2.0 * pi * std::sqrt(m / k);
    INFO("T_expected = " << T_expected);

    // Run one full period
    sim.run(T_expected, 0.0001);

    // After one period: q ≈ 0.1, q_dot ≈ 0
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.1, 0.001));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(0.0, 0.01));
}

TEST_CASE("PrismaticJoint: spring-mass energy conserved over many cycles",
          "[prismatic][spring][energy]")
{
    using namespace mbd;
    const Real m = 1.0;
    const Real k = 100.0;

    auto fx = make_slider(Vec3::UnitX(), m);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q(0) = 0.2;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
        tau(0) += -k * s.q(0);
    };

    auto compute_energy = [&]() -> Real {
        const Real KE = 0.5 * m * fx.sys.q_dot(0) * fx.sys.q_dot(0);
        const Real PE = 0.5 * k * fx.sys.q(0) * fx.sys.q(0);
        return KE + PE;
    };

    const Real E0 = compute_energy();

    // Run for 10 oscillation periods
    const Real T = 2.0 * pi * std::sqrt(m / k);
    sim.run(10.0 * T, 0.0001);

    const Real E1 = compute_energy();
    INFO("E0 = " << E0 << ", E1 = " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.005));
}

// ============================================================================
// Long-running stability
// ============================================================================

TEST_CASE("PrismaticJoint: constant velocity coast preserves velocity over 10s",
          "[prismatic][stability]")
{
    using namespace mbd;
    auto fx = make_slider(Vec3::UnitX());

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 3.0;
    fx.sys.compute_kinematics();

    sim.run(10.0, 0.001);

    // After 10s: q = 30, q_dot = 3
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(3.0, 1e-6));
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(30.0, 0.001));
}