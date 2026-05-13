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
    // ====================================================================
    // Double pendulum fixture
    //
    // Two equal-length rigid arms, each modeled as a point mass at the end
    // of a massless rod. Connected via revolute joints about world Z.
    //
    // Body 1: parent = ground, hinge at world origin.
    //         COM at body 1 frame (0, -L1, 0).
    // Body 2: parent = body 1, hinge at body-1 frame (0, -L1, 0)
    //         (i.e., at the COM of body 1).
    //         COM at body 2 frame (0, -L2, 0).
    // ====================================================================
    struct DoublePendulumFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body1{0};
        mbd::BodyIndex body2{0};
        mbd::Real m1{1.0};
        mbd::Real m2{1.0};
        mbd::Real L1{1.0};
        mbd::Real L2{1.0};
    };

    DoublePendulumFixture make_double_pendulum(mbd::Real m1 = 1.0,
                                                mbd::Real m2 = 1.0,
                                                mbd::Real L1 = 1.0,
                                                mbd::Real L2 = 1.0)
    {
        using namespace mbd;
        DoublePendulumFixture fx;
        fx.m1 = m1; fx.m2 = m2; fx.L1 = L1; fx.L2 = L2;

        // Body 1: point mass at distance L1 below body origin
        Mat3 I_com = Mat3::Identity() * Real(1e-6);
        RigidBodyInertia inertia1(m1, Vec3(0.0, -L1, 0.0), I_com);
        fx.body1 = fx.sys.add_body(inertia1, RigidBodyState{}, "arm1", kGroundIndex);

        // Joint 1: revolute about Z, hinge at world origin
        fx.sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body1));

        // Body 2: point mass at distance L2 below body origin
        RigidBodyInertia inertia2(m2, Vec3(0.0, -L2, 0.0), I_com);
        fx.body2 = fx.sys.add_body(inertia2, RigidBodyState{}, "arm2", fx.body1);

        // Joint 2: revolute about Z, hinge at body-1 frame (0, -L1, 0)
        // (the tip of arm 1, where arm 2 attaches)
        Transform3 X_PJ_2(Quat::Identity(), Vec3(0.0, -L1, 0.0));
        fx.sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            X_PJ_2, Transform3::Identity(),
            fx.body1, fx.body2));

        return fx;
    }

    // ====================================================================
    // Two-body chain: chassis on FreeCoord, wheel on prismatic
    // ====================================================================
    struct ChassisWheelFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex chassis{0};
        mbd::BodyIndex wheel{0};
        mbd::Real m_chassis{10.0};
        mbd::Real m_wheel{1.0};
    };

    ChassisWheelFixture make_chassis_wheel()
    {
        using namespace mbd;
        ChassisWheelFixture fx;

        // Chassis: solid box, 10 kg
        auto inertia_c = RigidBodyInertia::from_solid_box(fx.m_chassis,
                                                          Vec3(1.0, 0.5, 0.5));
        fx.chassis = fx.sys.add_body(inertia_c, RigidBodyState{}, "chassis",
                                      kGroundIndex);

        // FreeCoord joint to ground
        fx.sys.add_joint(std::make_unique<FreeCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.chassis));

        // Wheel: 1 kg
        auto inertia_w = RigidBodyInertia::from_solid_box(fx.m_wheel,
                                                          Vec3(0.2, 0.2, 0.2));
        fx.wheel = fx.sys.add_body(inertia_w, RigidBodyState{}, "wheel",
                                    fx.chassis);

        // Prismatic joint along world -Y at chassis frame (0, -0.5, 1.0)
        // Joint Z = world -Y, so increasing q moves wheel DOWN.
        // We set X_PJ to rotate joint Z to world -Y.
        Quat q_PJ = Quat::FromTwoVectors(Vec3::UnitZ(), -Vec3::UnitY());
        Transform3 X_PJ(q_PJ, Vec3(0.0, -0.5, 1.0));
        fx.sys.add_joint(std::make_unique<PrismaticCoordJoint>(
            X_PJ, Transform3::Identity(),
            fx.chassis, fx.wheel));

        return fx;
    }
}

// ============================================================================
// Two-body chain kinematics
// ============================================================================

TEST_CASE("Chain: double-pendulum at q=0 has both arms hanging straight",
          "[chain][kinematics]")
{
    using namespace mbd;
    auto fx = make_double_pendulum();

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    // Body 1 origin at world origin (hinge); COM at body-1 frame (0, -L1, 0)
    // which in world (with q=0, no rotation) is (0, -L1, 0) = (0, -1, 0).
    REQUIRE_THAT(fx.sys.states[fx.body1].p_WB.norm(), WithinAbs(0.0, 1e-12));

    // Body 2 origin at world (0, -L1, 0) (the tip of arm 1, where joint 2 is).
    REQUIRE_THAT(fx.sys.states[fx.body2].p_WB.x(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body2].p_WB.y(), WithinAbs(-fx.L1, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body2].p_WB.z(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Chain: double-pendulum with q1=pi/2 puts arm-1 horizontal",
          "[chain][kinematics]")
{
    using namespace mbd;
    auto fx = make_double_pendulum();

    // Rotate arm 1 by pi/2 about Z. Arm 1 was hanging in -Y; rotating by pi/2
    // (positive Z = right-hand rule) takes -Y to +X.
    fx.sys.q.setZero();
    fx.sys.q(0) = pi / 2.0;
    fx.sys.compute_kinematics();

    // Body 2 origin should be at the tip of arm 1: world (L1, 0, 0)
    REQUIRE_THAT(fx.sys.states[fx.body2].p_WB.x(), WithinAbs(fx.L1, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body2].p_WB.y(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Chain: double-pendulum velocity propagates",
          "[chain][kinematics]")
{
    using namespace mbd;
    auto fx = make_double_pendulum();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.0; // arm 1 rotating at 1 rad/s about Z
    fx.sys.compute_kinematics();

    // Body 1 angular velocity = 1 rad/s about Z
    REQUIRE_THAT(fx.sys.states[fx.body1].w_WB.z(), WithinAbs(1.0, 1e-9));

    // Body 2 angular velocity = body 1's (since q_dot(1)=0)
    REQUIRE_THAT(fx.sys.states[fx.body2].w_WB.z(), WithinAbs(1.0, 1e-9));

    // Body 2 origin is at tip of arm 1, world (0, -L1, 0).
    // Velocity of tip of arm 1 = omega × r = (1 about Z) × (0, -L1, 0) = (L1, 0, 0)
    REQUIRE_THAT(fx.sys.states[fx.body2].v_WB.x(), WithinAbs(fx.L1, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body2].v_WB.y(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// Two-body chain dynamics
// ============================================================================

TEST_CASE("Chain: double-pendulum at rest hangs straight down",
          "[chain][equilibrium]")
{
    using namespace mbd;
    auto fx = make_double_pendulum();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q(1), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q_dot(1), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Chain: double-pendulum energy conserved during chaotic motion",
          "[chain][energy]")
{
    using namespace mbd;
    const Real L1 = 1.0;
    const Real L2 = 1.0;
    const Real m1 = 1.0;
    const Real m2 = 1.0;
    auto fx = make_double_pendulum(m1, m2, L1, L2);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Strong initial displacement to get full nonlinear chaotic motion
    fx.sys.q.setZero();
    fx.sys.q(0) = 1.0;
    fx.sys.q(1) = 0.5;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        // KE of each body's COM
        Real KE = 0.0;
        for (BodyIndex b : { fx.body1, fx.body2 }) {
            const Vec3 w = fx.sys.states[b].w_WB;
            const Vec3 com_offset_W =
                fx.sys.states[b].q_WB * fx.sys.inertias[b].com_B;
            const Vec3 v_com_W = fx.sys.states[b].v_WB + w.cross(com_offset_W);
            const Real m = fx.sys.inertias[b].mass;
            KE += 0.5 * m * v_com_W.squaredNorm();
        }

        // PE of each body's COM
        Real PE = 0.0;
        for (BodyIndex b : { fx.body1, fx.body2 }) {
            const Vec3 com_W = fx.sys.states[b].p_WB +
                               fx.sys.states[b].q_WB * fx.sys.inertias[b].com_B;
            const Real m = fx.sys.inertias[b].mass;
            PE += m * g_accel * com_W.y();
        }

        return KE + PE;
    };

    const Real E0 = compute_energy();

    // Run 5 seconds — chaotic motion happens here
    sim.run(5.0, 0.0001);

    const Real E1 = compute_energy();
    INFO("E0 = " << E0 << ", E1 = " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    // Allow 1% drift over 5s with RK4. Chaotic motion is sensitive to step size.
    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.01));
}

TEST_CASE("Chain: double-pendulum small-angle in-phase mode",
          "[chain][modes]")
{
    using namespace mbd;
    // Equal-mass, equal-length double pendulum has two normal modes.
    // The slower (in-phase) mode has both arms swinging together.
    // Frequency: omega_slow = sqrt((2 - sqrt(2)) * g / L)  for m1=m2, L1=L2=L
    const Real L = 1.0;
    auto fx = make_double_pendulum(1.0, 1.0, L, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // For in-phase (slow) mode: in ABSOLUTE angles (theta1, theta2), the
    // eigenvector is (1, sqrt(2)). Our parameterization uses RELATIVE angles
    // (q0, q1) where q0 = theta1, q1 = theta2 - theta1. So mode shape becomes
    // (q0, q1) = (1, sqrt(2) - 1).
    fx.sys.q.setZero();
    fx.sys.q(0) = 0.05;
    fx.sys.q(1) = 0.05 * (std::sqrt(2.0) - 1.0);
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real omega_slow = std::sqrt((2.0 - std::sqrt(2.0)) * g_accel / L);
    const Real T_slow = 2.0 * pi / omega_slow;

    INFO("T_slow = " << T_slow);

    sim.run(T_slow, 0.0001);

    // After one period, both DOFs should be back to initial
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.05, 0.005));
    REQUIRE_THAT(fx.sys.q(1), WithinAbs(0.05 * (std::sqrt(2.0) - 1.0), 0.005));
}

// ============================================================================
// Three-body chain: chassis with prismatic wheel
// ============================================================================

TEST_CASE("Chain: chassis-wheel kinematic offsets at rest",
          "[chain][chassis_wheel]")
{
    using namespace mbd;
    auto fx = make_chassis_wheel();

    // q[0:6] for FreeCoord chassis, q[6] for wheel prismatic
    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    // Chassis at world origin, identity rotation
    REQUIRE_THAT(fx.sys.states[fx.chassis].p_WB.norm(), WithinAbs(0.0, 1e-12));

    // Wheel: prismatic at chassis-frame (0, -0.5, 1.0), q=0 means no extra
    // displacement. Wheel origin in chassis frame = (0, -0.5, 1.0).
    // In world frame (chassis at origin, identity rot): same.
    REQUIRE_THAT(fx.sys.states[fx.wheel].p_WB.y(), WithinAbs(-0.5, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.wheel].p_WB.z(), WithinAbs(1.0, 1e-12));
}

TEST_CASE("Chain: off-center wheel induces chassis rotation under chassis-frame force",
          "[chain][chassis_wheel]")
{
    using namespace mbd;
    // Use the original off-center wheel fixture
    auto fx = make_chassis_wheel();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Apply force at chassis origin. Chassis COM = chassis origin (com_B=0 for
    // solid box). Wheel at chassis-frame (0, -0.5, 1.0) with mass 1.
    // System COM = (m_c·(0,0,0) + m_w·(0,-0.5,1))/(m_c+m_w) = (0, -1/22, 1/11).
    // Force at chassis origin is OFFSET from system COM in Z by 1/11 m,
    // so it induces a small torque about chassis Y. Chassis rotates a bit.
    const Real F = 11.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    // Verify physics:
    //   1. SOME rotation about Y is induced
    REQUIRE(std::abs(fx.sys.q(4)) > 1e-3);

    //   2. The system center of mass translates as F=ma predicts:
    //      a_com = F/m_total = 1, so com.x = 0.5 after 1s
    const Vec3 chassis_com_W = fx.sys.states[fx.chassis].p_WB; // com_B = 0
    const Vec3 wheel_com_W   = fx.sys.states[fx.wheel].p_WB;   // com_B = 0
    const Real m_total = fx.m_chassis + fx.m_wheel;
    const Vec3 sys_com_W = (fx.m_chassis * chassis_com_W + fx.m_wheel * wheel_com_W)
                          / m_total;

    INFO("system COM x = " << sys_com_W.x());
    INFO("chassis q(4) = " << fx.sys.q(4));

    REQUIRE_THAT(sys_com_W.x(), WithinAbs(0.5, 0.01));
}

TEST_CASE("Chain: chassis acceleration translates wheel correctly",
          "[chain][chassis_wheel]")
{
    using namespace mbd;
    // For this test, build a chassis-wheel system with the wheel at the
    // chassis ORIGIN (no offset). This isolates pure F=ma behavior without
    // the off-center coupling that would otherwise induce chassis rotation.
    MultibodySystem sys;

    const Real m_chassis = 10.0;
    const Real m_wheel   = 1.0;

    auto inertia_c = RigidBodyInertia::from_solid_box(m_chassis,
                                                       Vec3(1.0, 0.5, 0.5));
    BodyIndex chassis = sys.add_body(inertia_c, RigidBodyState{}, "chassis",
                                      kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, chassis));

    auto inertia_w = RigidBodyInertia::from_solid_box(m_wheel,
                                                       Vec3(0.2, 0.2, 0.2));
    BodyIndex wheel = sys.add_body(inertia_w, RigidBodyState{}, "wheel",
                                    chassis);

    // Prismatic at chassis origin, joint Z = world -Y
    Quat q_PJ = Quat::FromTwoVectors(Vec3::UnitZ(), -Vec3::UnitY());
    Transform3 X_PJ(q_PJ, Vec3::Zero()); // wheel at chassis origin
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_PJ, Transform3::Identity(),
        chassis, wheel));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.compute_kinematics();

    // Force F = 11N applied to chassis tau(0). Total mass = 11 kg, no offset
    // so no rotation coupling. Expected: a = 1 m/s², after 1s: p = 0.5 m.
    const Real F = 11.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    REQUIRE_THAT(sys.states[chassis].p_WB.x(), WithinAbs(0.5, 0.005));
    REQUIRE_THAT(sys.states[wheel].p_WB.x(),   WithinAbs(0.5, 0.005));

    // Both should be at exactly the same X position
    REQUIRE_THAT(sys.states[chassis].p_WB.x() - sys.states[wheel].p_WB.x(),
                 WithinAbs(0.0, 1e-9));

    // No rotation should be induced
    REQUIRE_THAT(sys.q(3), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(sys.q(4), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(sys.q(5), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Chain: wheel free to fall under gravity even when chassis is fixed",
          "[chain][chassis_wheel][gravity]")
{
    using namespace mbd;
    auto fx = make_chassis_wheel();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Hold chassis in place via velocity-feedback force on its DOFs (so we
    // can isolate the wheel's behavior).
    sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
        // Strongly damp all chassis DOFs (q_dot[0:6])
        for (int i = 0; i < 6; ++i) {
            tau(i) -= 1000.0 * s.q_dot(i);
        }
    };

    sim.run(0.5, 0.001);

    // Wheel's prismatic joint Z is aligned with world -Y, so when wheel falls
    // under gravity, q (the wheel's prismatic DOF, index 6) should INCREASE.
    // After 0.5s of free fall with gravity g: q = 0.5 * g * 0.5^2 = 1.226
    const Real expected_q = 0.5 * g_accel * 0.5 * 0.5;
    REQUIRE_THAT(fx.sys.q(6), WithinAbs(expected_q, expected_q * 0.05));
}

TEST_CASE("Chain: wheel position tracks chassis under translation",
          "[chain][chassis_wheel]")
{
    using namespace mbd;
    auto fx = make_chassis_wheel();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Pre-translate chassis by setting q(0:3)
    fx.sys.q.setZero();
    fx.sys.q(0) = 5.0;  // chassis at world (5, 0, 0)
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Wheel should be at chassis-frame (0, -0.5, 1.0) translated by (5, 0, 0)
    // = world (5, -0.5, 1.0)
    REQUIRE_THAT(fx.sys.states[fx.wheel].p_WB.x(), WithinAbs(5.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.wheel].p_WB.y(), WithinAbs(-0.5, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.wheel].p_WB.z(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("Chain: wheel velocity tracks chassis velocity",
          "[chain][chassis_wheel]")
{
    using namespace mbd;
    auto fx = make_chassis_wheel();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 3.0;  // chassis moving in +X at 3 m/s
    fx.sys.compute_kinematics();

    // Wheel's velocity = chassis velocity (no extra wheel motion since q_dot[6]=0)
    REQUIRE_THAT(fx.sys.states[fx.wheel].v_WB.x(), WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.wheel].v_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.wheel].v_WB.z(), WithinAbs(0.0, 1e-9));
}