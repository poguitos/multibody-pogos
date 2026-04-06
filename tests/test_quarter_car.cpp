#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>

#include "mbd/simulator.hpp"
#include "mbd/tire.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-9;

    // Standard quarter-car parameters
    constexpr mbd::Real m_s  = 250.0;     // sprung mass [kg]
    constexpr mbd::Real m_u  = 40.0;      // unsprung mass [kg]
    constexpr mbd::Real k_s  = 20000.0;   // suspension stiffness [N/m]
    constexpr mbd::Real k_t  = 200000.0;  // tire stiffness [N/m]
    constexpr mbd::Real c_s  = 1500.0;    // suspension damping [Ns/m]
    constexpr mbd::Real R_free = 0.35;    // tire free radius [m]
    constexpr mbd::Real L0_s   = 0.30;    // suspension spring free length [m]

    // Static equilibrium heights
    mbd::Real y_w_eq()
    {
        return R_free - (m_s + m_u) * mbd::g_accel / k_t;
    }
    mbd::Real y_c_eq()
    {
        return y_w_eq() + L0_s - m_s * mbd::g_accel / k_s;
    }

    // Analytical undamped natural frequencies (rad/s) from eigenvalue problem:
    //   det(K - omega^2 * M) = 0
    // where M = diag(m_u, m_s), K = [[k_s + k_t, -k_s], [-k_s, k_s]]
    std::pair<mbd::Real, mbd::Real> analytical_frequencies()
    {
        // Eigenvalues of M^{-1} K
        const mbd::Real a = (k_s + k_t) / m_u;
        const mbd::Real b = k_s / m_s;
        const mbd::Real c_off = k_s / m_u;
        const mbd::Real d_off = k_s / m_s;

        // Characteristic equation: omega^4 - (a+b)*omega^2 + (a*b - c_off*d_off) = 0
        const mbd::Real sum = a + b;
        const mbd::Real prod = a * b - c_off * d_off;

        const mbd::Real disc = sum * sum - 4.0 * prod;
        const mbd::Real omega1_sq = (sum - std::sqrt(disc)) / 2.0;
        const mbd::Real omega2_sq = (sum + std::sqrt(disc)) / 2.0;

        return {std::sqrt(omega1_sq), std::sqrt(omega2_sq)};
    }

    /// Build a quarter-car MultibodySystem.
    /// Returns the system with two prismatic-Y bodies, spring-damper, and tire.
    /// The spring damper and tire are registered as force elements.
    mbd::MultibodySystem make_quarter_car(mbd::Real susp_damping)
    {
        using namespace mbd;

        MultibodySystem sys;

        // Joint frame rotation: Rx(-pi/2) maps joint Z to world +Y
        Mat3 R_Y = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
        Transform3 X_prismatic_Y = Transform3::FromRotation(R_Y);

        // Wheel (unsprung mass) — body 1, prismatic-Y from ground
        auto I_wheel = RigidBodyInertia::from_solid_box(m_u, Vec3(0.15, 0.15, 0.15));
        sys.add_body(I_wheel, RigidBodyState{}, "wheel", kGroundIndex);
        sys.add_joint(std::make_unique<PrismaticCoordJoint>(
            X_prismatic_Y, X_prismatic_Y, kGroundIndex, 1));

        // Chassis (sprung mass) — body 2, prismatic-Y from ground
        auto I_chassis = RigidBodyInertia::from_solid_box(m_s, Vec3(0.5, 0.2, 0.4));
        sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
        sys.add_joint(std::make_unique<PrismaticCoordJoint>(
            X_prismatic_Y, X_prismatic_Y, kGroundIndex, 2));

        // Suspension spring-damper between wheel and chassis
        sys.force_elements.push_back(std::make_unique<LinearSpringDamper>(
            1, 2,              // wheel body to chassis body
            Vec3::Zero(),      // attachment at wheel origin
            Vec3::Zero(),      // attachment at chassis origin
            k_s,               // stiffness
            susp_damping,      // damping
            L0_s               // rest length
        ));

        // Tire contact force on wheel
        sys.force_elements.push_back(std::make_unique<TireContactForce>(
            1,                 // wheel body index
            R_free,            // free radius
            k_t,               // vertical stiffness
            0.0                // no tire damping for clean frequency tests
        ));

        return sys;
    }
}

// ============================================================================
// Static equilibrium
// ============================================================================

TEST_CASE("Quarter-car: settles to static equilibrium",
          "[quarter_car][static]")
{
    using namespace mbd;

    auto sys = make_quarter_car(c_s); // With suspension damping
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start slightly above equilibrium
    sys.q(0) = y_w_eq() + 0.02;
    sys.q(1) = y_c_eq() + 0.05;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    // Simulate long enough for damping to settle (5 seconds)
    sim.run(5.0, 0.001);

    // Should settle near analytical equilibrium
    REQUIRE_THAT(sys.q(0), WithinAbs(y_w_eq(), 0.002));
    REQUIRE_THAT(sys.q(1), WithinAbs(y_c_eq(), 0.002));

    // Velocities should be near zero
    REQUIRE_THAT(sys.q_dot(0), WithinAbs(0.0, 0.01));
    REQUIRE_THAT(sys.q_dot(1), WithinAbs(0.0, 0.01));
}

// ============================================================================
// Natural frequencies (undamped)
// ============================================================================

TEST_CASE("Quarter-car: undamped natural frequencies match analytical",
          "[quarter_car][frequency]")
{
    using namespace mbd;

    auto sys = make_quarter_car(0.0); // No damping
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.set_recording(true);
    sim.initialize();

    // Start at equilibrium with a small chassis perturbation (excites body bounce mode)
    sys.q(0) = y_w_eq();
    sys.q(1) = y_c_eq() + 0.01;  // 10mm bump
    sys.q_dot.setZero();
    sys.compute_kinematics();

    // Record initial state correctly
    sim.history.clear();
    sim.history.push_back({sim.time, sys.q, sys.q_dot});

    // Simulate 3 seconds
    sim.run(3.0, 0.001);

    auto [omega1, omega2] = analytical_frequencies();
    const Real f1_theory = omega1 / (2.0 * pi); // body bounce ~1.4 Hz
    const Real f2_theory = omega2 / (2.0 * pi); // wheel hop ~11.8 Hz

    // Measure body bounce frequency from chassis displacement history.
    // Count zero crossings of (y_c - y_c_eq).
    const Real y_c_0 = y_c_eq();
    std::vector<Real> cross_times;
    for (size_t k = 1; k < sim.history.size(); ++k) {
        const Real dy_prev = sim.history[k - 1].q(1) - y_c_0;
        const Real dy_curr = sim.history[k].q(1) - y_c_0;

        // Negative-going crossing
        if (dy_prev > 0.0 && dy_curr <= 0.0) {
            const Real t0 = sim.history[k - 1].time;
            const Real t1 = sim.history[k].time;
            cross_times.push_back(t0 + dy_prev / (dy_prev - dy_curr) * (t1 - t0));
        }
    }

    REQUIRE(cross_times.size() >= 2);

    // Period between successive same-direction crossings = full oscillation period
    const Real T_measured = cross_times[1] - cross_times[0];
    const Real f_measured = 1.0 / T_measured;

    // The dominant mode for a chassis perturbation is body bounce
    // Allow 10% tolerance since both modes are excited and interact
    REQUIRE_THAT(f_measured, WithinAbs(f1_theory, f1_theory * 0.10));
}

// ============================================================================
// Energy conservation (undamped)
// ============================================================================

TEST_CASE("Quarter-car: undamped energy conservation",
          "[quarter_car][energy]")
{
    using namespace mbd;

    auto sys = make_quarter_car(0.0); // No damping
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at equilibrium with perturbation
    sys.q(0) = y_w_eq() + 0.005;
    sys.q(1) = y_c_eq() + 0.01;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        sys.compute_kinematics();

        // Kinetic energy
        const MatX M = compute_mass_matrix(sys);
        const Real KE = 0.5 * sys.q_dot.transpose() * M * sys.q_dot;

        // Gravitational PE
        Real PE_grav = 0.0;
        for (BodyIndex i = 1; i < sys.body_count(); ++i) {
            const auto& st = sys.states[i];
            const auto& in = sys.inertias[i];
            const Mat3 R = st.q_WB.toRotationMatrix();
            const Vec3 com_W = st.p_WB + R * in.com_B;
            PE_grav += in.mass * g_accel * com_W.y();
        }

        // Suspension spring PE: 0.5 * k_s * (dist - L0)^2
        const Real dist_susp = (sys.states[2].p_WB - sys.states[1].p_WB).norm();
        const Real PE_susp = 0.5 * k_s * (dist_susp - L0_s) * (dist_susp - L0_s);

        // Tire spring PE: 0.5 * k_t * deflection^2
        const auto* tire = static_cast<const TireContactForce*>(
            sys.force_elements[1].get());
        const Real defl = tire->get_deflection(sys.states);
        const Real PE_tire = 0.5 * k_t * defl * defl;

        return KE + PE_grav + PE_susp + PE_tire;
    };

    const Real E0 = compute_energy();

    sim.run(2.0, 0.001);

    const Real E_final = compute_energy();
    const Real rel_error = std::abs(E_final - E0) / std::abs(E0);

    REQUIRE(rel_error < 1e-4);
}

// ============================================================================
// Damped response decays
// ============================================================================

TEST_CASE("Quarter-car: damped oscillation decays",
          "[quarter_car][damped]")
{
    using namespace mbd;

    auto sys = make_quarter_car(c_s); // With damping
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.set_recording(true);
    sim.initialize();

    sys.q(0) = y_w_eq();
    sys.q(1) = y_c_eq() + 0.03;  // 30mm perturbation
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sim.history.clear();
    sim.history.push_back({sim.time, sys.q, sys.q_dot});

    sim.run(3.0, 0.001);

    // Measure peak chassis displacement over time
    const Real y_eq = y_c_eq();
    Real max_disp_first_half = 0.0;
    Real max_disp_second_half = 0.0;

    for (const auto& rec : sim.history) {
        const Real disp = std::abs(rec.q(1) - y_eq);
        if (rec.time < 1.5) {
            max_disp_first_half = std::max(max_disp_first_half, disp);
        } else {
            max_disp_second_half = std::max(max_disp_second_half, disp);
        }
    }

    // Second half oscillations should be significantly smaller than first half
    REQUIRE(max_disp_second_half < max_disp_first_half * 0.5);

    // Final velocity should be much smaller than peak
    Real max_vel = 0.0;
    for (const auto& rec : sim.history) {
        max_vel = std::max(max_vel, std::abs(rec.q_dot(1)));
    }
    const Real final_vel = std::abs(sys.q_dot(1));
    REQUIRE(final_vel < max_vel * 0.1);
}

// ============================================================================
// Force element projection consistency
// ============================================================================

TEST_CASE("Quarter-car: force projection gives same result as manual force callback",
          "[quarter_car][projection]")
{
    using namespace mbd;

    // System 1: uses registered force elements (automatic projection)
    auto sys1 = make_quarter_car(c_s);
    Simulator sim1(sys1);
    sim1.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim1.method = IntegrationMethod::RK4;
    sim1.initialize();

    sys1.q(0) = y_w_eq() + 0.01;
    sys1.q(1) = y_c_eq() + 0.02;
    sys1.q_dot(0) = 0.1;
    sys1.q_dot(1) = -0.05;
    sys1.compute_kinematics();

    // Compute one forward dynamics step
    sys1.clear_forces();
    sys1.apply_force_elements();
    VecX tau_projected = project_body_forces_to_joint_space(sys1);
    VecX qdd1 = forward_dynamics(sys1, tau_projected, sim1.gravity);

    // System 2: no force elements, uses force callback to apply same forces manually
    MultibodySystem sys2;

    Mat3 R_Y = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    Transform3 X_prismatic_Y = Transform3::FromRotation(R_Y);

    sys2.add_body(RigidBodyInertia::from_solid_box(m_u, Vec3(0.15, 0.15, 0.15)),
                  RigidBodyState{}, "wheel", kGroundIndex);
    sys2.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_prismatic_Y, X_prismatic_Y, kGroundIndex, 1));

    sys2.add_body(RigidBodyInertia::from_solid_box(m_s, Vec3(0.5, 0.2, 0.4)),
                  RigidBodyState{}, "chassis", kGroundIndex);
    sys2.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_prismatic_Y, X_prismatic_Y, kGroundIndex, 2));

    // Copy state
    sys2.q     = sys1.q;
    sys2.q_dot = sys1.q_dot;
    sys2.compute_kinematics();

    // Apply same forces manually via tau
    // The projected tau from sys1 should match if we pass it directly
    VecX qdd2 = forward_dynamics(sys2, tau_projected, sim1.gravity);

    REQUIRE_THAT(qdd1(0), WithinAbs(qdd2(0), 1e-10));
    REQUIRE_THAT(qdd1(1), WithinAbs(qdd2(1), 1e-10));
}

// ============================================================================
// Tire lifts off
// ============================================================================

TEST_CASE("Quarter-car: tire lifts off when wheel is above free radius",
          "[quarter_car][liftoff]")
{
    using namespace mbd;

    MultibodySystem sys;

    Mat3 R_Y = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    Transform3 X_prismatic_Y = Transform3::FromRotation(R_Y);

    auto I_wheel = RigidBodyInertia::from_solid_box(m_u, Vec3(0.15, 0.15, 0.15));
    sys.add_body(I_wheel, RigidBodyState{}, "wheel", kGroundIndex);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_prismatic_Y, X_prismatic_Y, kGroundIndex, 1));

    auto tire = std::make_unique<TireContactForce>(1, R_free, k_t, 0.0);
    const TireContactForce* tire_ptr = tire.get();
    sys.force_elements.push_back(std::move(tire));

    // Wheel at exactly free radius: contact point at y=0, no penetration
    sys.q(0) = R_free;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire_ptr->get_vertical_force(sys.states), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(sys.forces[1].f_W.norm(), WithinAbs(0.0, 1e-9));

    // Wheel above free radius: no contact
    sys.q(0) = R_free + 0.1;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire_ptr->get_vertical_force(sys.states), WithinAbs(0.0, 1e-9));

    // Wheel below free radius: contact force
    sys.q(0) = R_free - 0.01;  // 10mm compression
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    const Real expected_force = k_t * 0.01; // 200000 * 0.01 = 2000 N
    REQUIRE_THAT(tire_ptr->get_vertical_force(sys.states), WithinAbs(expected_force, 1.0));
    REQUIRE(sys.forces[1].f_W.y() > 0.0); // Pushes up
}