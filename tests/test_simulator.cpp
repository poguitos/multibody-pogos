#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps_coarse = 1e-3;
    constexpr mbd::Real eps_fine   = 1e-6;

    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }

    /// Build a single pendulum (1m bar, 1 kg, pivot at origin, axis Z).
    mbd::MultibodySystem make_pendulum()
    {
        using namespace mbd;
        MultibodySystem sys;
        auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
        sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);
        sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::Identity(),
            Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
            kGroundIndex, 1));
        return sys;
    }

    /// Build a prismatic slider along Y (mass on a vertical rail).
    mbd::MultibodySystem make_vertical_slider(mbd::Real mass)
    {
        using namespace mbd;
        MultibodySystem sys;
        auto inertia = RigidBodyInertia::from_solid_box(mass, Vec3(0.1, 0.1, 0.1));
        sys.add_body(inertia, RigidBodyState{}, "slider", kGroundIndex);

        // Prismatic joint axis Z, rotated so Z aligns with world Y: Rx(-pi/2)
        Mat3 R = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
        Transform3 X_J_frame = Transform3::FromRotation(R);
        sys.add_joint(std::make_unique<PrismaticCoordJoint>(
            X_J_frame, X_J_frame, kGroundIndex, 1));
        return sys;
    }
}

// ============================================================================
// Prismatic free fall (exact analytical solution)
// ============================================================================

TEST_CASE("Simulator: prismatic free fall under gravity (RK4)",
          "[simulator][rk4]")
{
    using namespace mbd;

    auto sys = make_vertical_slider(2.0);
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.set_recording(true);
    sim.initialize();

    // Start at y=10, zero velocity
    sys.q(0) = 10.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    const Real dt = 0.001;
    const Real T  = 1.0;
    sim.run(T, dt);

    // Analytical: y = y0 + v0*t - 0.5*g*t^2 = 10 - 4.905
    const Real y_expected = 10.0 - 0.5 * g_accel * T * T;
    const Real v_expected = -g_accel * T;

    REQUIRE_THAT(sys.q(0), WithinAbs(y_expected, 1e-6));
    REQUIRE_THAT(sys.q_dot(0), WithinAbs(v_expected, 1e-6));

    // Position in world frame
    require_vec3_near(sys.states[1].p_WB, Vec3(0.0, y_expected, 0.0), 1e-5);
}

TEST_CASE("Simulator: prismatic free fall under gravity (Semi-implicit Euler)",
          "[simulator][euler]")
{
    using namespace mbd;

    auto sys = make_vertical_slider(2.0);
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::SemiImplicitEuler;
    sim.initialize();

    sys.q(0) = 10.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    const Real dt = 0.001;
    const Real T  = 1.0;
    sim.run(T, dt);

    const Real y_expected = 10.0 - 0.5 * g_accel * T * T;
    const Real v_expected = -g_accel * T;

    // Euler is first-order: expect ~O(dt) error, so ~1e-3 tolerance
    REQUIRE_THAT(sys.q(0), WithinAbs(y_expected, 0.02));
    REQUIRE_THAT(sys.q_dot(0), WithinAbs(v_expected, 0.02));
}

// ============================================================================
// Energy conservation (pendulum)
// ============================================================================

TEST_CASE("Simulator: pendulum energy conservation with RK4",
          "[simulator][rk4][energy]")
{
    using namespace mbd;

    auto sys = make_pendulum();
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at 45 degrees, zero velocity
    sys.q(0) = pi / 4.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    // Compute initial energy
    auto compute_energy = [&]() -> Real {
        sys.compute_kinematics();
        const MatX M = compute_mass_matrix(sys);
        const Real KE = 0.5 * sys.q_dot.transpose() * M * sys.q_dot;

        // PE = m * g * y_com
        const auto& state = sys.states[1];
        const auto& inertia = sys.inertias[1];
        const Mat3 R_WB = state.q_WB.toRotationMatrix();
        const Vec3 com_W = state.p_WB + R_WB * inertia.com_B;
        const Real PE = inertia.mass * g_accel * com_W.y();

        return KE + PE;
    };

    const Real E0 = compute_energy();

    // Simulate for 2 seconds (several oscillation periods)
    const Real dt = 0.0005;
    const Real T  = 2.0;
    sim.run(T, dt);

    const Real E_final = compute_energy();

    // RK4 with small dt should conserve energy to ~1e-5 relative
    const Real rel_error = std::abs(E_final - E0) / std::abs(E0);
    REQUIRE(rel_error < 1e-5);
}

// ============================================================================
// Small-angle pendulum period
// ============================================================================

TEST_CASE("Simulator: small-angle pendulum period matches theory",
          "[simulator][rk4][period]")
{
    using namespace mbd;

    auto sys = make_pendulum();
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.set_recording(true);
    sim.initialize();

    // Small angle about the stable equilibrium (hanging down = -pi/2)
    const Real q_eq = -pi / 2.0;
    sys.q(0) = q_eq + 0.05;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    // Theoretical period: T = 2*pi * sqrt(I_pivot / (m*g*d))
    const Real I_zz = (1.0 / 3.0) * (0.25 + 0.0025);
    const Real I_pivot = I_zz + 0.25;
    const Real d_com = 0.5;
    const Real T_theory = 2.0 * pi * std::sqrt(I_pivot / (1.0 * g_accel * d_com));

    // Simulate for 3 full periods with dt=0.001 (RK4 is accurate enough)
    const Real dt = 0.001;
    sim.run(3.0 * T_theory, dt);

    // Detect crossings of q = q_eq going negative (q drops below equilibrium)
    std::vector<Real> zero_cross_times;
    for (size_t k = 1; k < sim.history.size(); ++k) {
        const Real dq_prev = sim.history[k - 1].q(0) - q_eq;
        const Real dq_curr = sim.history[k].q(0) - q_eq;

        // Negative-going crossing of equilibrium
        if (dq_prev > 0.0 && dq_curr <= 0.0) {
            const Real t_prev = sim.history[k - 1].time;
            const Real t_curr = sim.history[k].time;
            const Real t_cross = t_prev + dq_prev / (dq_prev - dq_curr) * (t_curr - t_prev);
            zero_cross_times.push_back(t_cross);
        }
    }

    // Should have at least 3 crossings in 3 periods
    REQUIRE(zero_cross_times.size() >= 3);

    // Period = time between successive same-direction crossings
    const Real T_measured = zero_cross_times[1] - zero_cross_times[0];

    REQUIRE_THAT(T_measured, WithinAbs(T_theory, 1e-3));
}

// ============================================================================
// Force callback
// ============================================================================

TEST_CASE("Simulator: force callback applies joint torque",
          "[simulator][callback]")
{
    using namespace mbd;

    auto sys = make_pendulum();
    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero()); // No gravity
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sys.q(0) = 0.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    // Apply constant torque of 1.0 N*m
    const Real tau_const = 1.0;
    sim.force_callback = [tau_const](MultibodySystem&, Real, VecX& tau) {
        tau(0) += tau_const;
    };

    const Real dt = 0.001;
    const Real T  = 1.0;
    sim.run(T, dt);

    // q_ddot = tau / I_pivot (constant)
    const Real I_zz = (1.0 / 3.0) * (0.25 + 0.0025);
    const Real I_pivot = I_zz + 0.25;
    const Real q_ddot = tau_const / I_pivot;

    const Real q_expected  = 0.5 * q_ddot * T * T;
    const Real qd_expected = q_ddot * T;

    REQUIRE_THAT(sys.q(0), WithinAbs(q_expected, 1e-5));
    REQUIRE_THAT(sys.q_dot(0), WithinAbs(qd_expected, 1e-5));
}

// ============================================================================
// Double pendulum basic sanity
// ============================================================================

TEST_CASE("Simulator: double pendulum runs without crash and conserves energy",
          "[simulator][rk4][double_pendulum]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
    sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);
    sys.add_body(inertia, RigidBodyState{}, "link2", 1);

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, 1));
    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0)),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        1, 2));

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sys.q << 0.5, -0.3;
    sys.q_dot << 0.0, 0.0;
    sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        sys.compute_kinematics();
        const MatX M = compute_mass_matrix(sys);
        const Real KE = 0.5 * sys.q_dot.transpose() * M * sys.q_dot;

        Real PE = 0.0;
        for (BodyIndex i = 1; i < sys.body_count(); ++i) {
            const auto& st = sys.states[i];
            const auto& in = sys.inertias[i];
            const Mat3 R = st.q_WB.toRotationMatrix();
            const Vec3 com_W = st.p_WB + R * in.com_B;
            PE += in.mass * g_accel * com_W.y();
        }
        return KE + PE;
    };

    const Real E0 = compute_energy();

    // 3 seconds of chaotic double pendulum
    sim.run(3.0, 0.0005);

    const Real E_final = compute_energy();
    const Real rel_error = std::abs(E_final - E0) / std::abs(E0);

    // RK4 with dt=0.5ms should keep energy drift under 0.01%
    REQUIRE(rel_error < 1e-4);

    // Sanity: system should have moved (not stuck)
    REQUIRE(std::abs(sys.q(0)) > 0.01);
    REQUIRE(std::abs(sys.q_dot(0)) > 0.01);
}

// ============================================================================
// RK4 vs Euler convergence
// ============================================================================

TEST_CASE("Simulator: RK4 is more accurate than Euler at same step size",
          "[simulator][convergence]")
{
    using namespace mbd;

    const Real dt = 0.01;
    const Real T  = 0.5;
    const Vec3 grav(0.0, -g_accel, 0.0);

    // Reference: RK4 with very small dt
    auto sys_ref = make_pendulum();
    Simulator sim_ref(sys_ref);
    sim_ref.set_gravity(grav);
    sim_ref.method = IntegrationMethod::RK4;
    sim_ref.initialize();
    sys_ref.q(0) = 0.5;
    sys_ref.q_dot(0) = 0.0;
    sys_ref.compute_kinematics();
    sim_ref.run(T, 0.0001);
    const Real q_ref = sys_ref.q(0);

    // RK4 at dt = 0.01
    auto sys_rk4 = make_pendulum();
    Simulator sim_rk4(sys_rk4);
    sim_rk4.set_gravity(grav);
    sim_rk4.method = IntegrationMethod::RK4;
    sim_rk4.initialize();
    sys_rk4.q(0) = 0.5;
    sys_rk4.q_dot(0) = 0.0;
    sys_rk4.compute_kinematics();
    sim_rk4.run(T, dt);
    const Real err_rk4 = std::abs(sys_rk4.q(0) - q_ref);

    // Euler at dt = 0.01
    auto sys_euler = make_pendulum();
    Simulator sim_euler(sys_euler);
    sim_euler.set_gravity(grav);
    sim_euler.method = IntegrationMethod::SemiImplicitEuler;
    sim_euler.initialize();
    sys_euler.q(0) = 0.5;
    sys_euler.q_dot(0) = 0.0;
    sys_euler.compute_kinematics();
    sim_euler.run(T, dt);
    const Real err_euler = std::abs(sys_euler.q(0) - q_ref);

    // RK4 should be orders of magnitude better
    REQUIRE(err_rk4 < err_euler * 0.01);
}

// ============================================================================
// State recording
// ============================================================================

TEST_CASE("Simulator: recording captures correct number of snapshots",
          "[simulator][recording]")
{
    using namespace mbd;

    auto sys = make_pendulum();
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.set_recording(true);
    sim.initialize();

    sys.q(0) = 0.3;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    const Real dt = 0.01;
    const int steps = 100;
    sim.run(steps * dt, dt);

    // Initial state + 100 steps = 101 records
    REQUIRE(sim.history.size() == 101);

    REQUIRE_THAT(sim.history.front().time, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(sim.history.back().time, WithinAbs(1.0, 1e-9));

    // Each record has correct size
    for (const auto& rec : sim.history) {
        REQUIRE(rec.q.size() == 1);
        REQUIRE(rec.q_dot.size() == 1);
    }
}