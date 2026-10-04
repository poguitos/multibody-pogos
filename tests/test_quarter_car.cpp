#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>
#include <vector>

#include "mbd/forces/tire.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/simulator.hpp"

// The quarter car on the kernel (plan task 2.7): two bodies sliding vertically,
// a suspension spring-damper between them and a tyre under the wheel.

using Catch::Matchers::WithinAbs;

namespace
{
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

    /// The two bodies on prismatic joints along the world Y axis: the wheel
    /// (body 1, coordinate 0) and the chassis (body 2, coordinate 1).
    void add_quarter_car_bodies(mbd::kernel::System& sys)
    {
        using namespace mbd;
        sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
        // Joint frame rotation: Rx(-pi/2) maps joint Z to world +Y
        const Transform3 X_prismatic_Y = Transform3::FromRotation(
            Mat3(Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix()));
        const auto prismatic = std::make_shared<kernel::PrismaticJointModel>();
        sys.model.add_body(0, prismatic, X_prismatic_Y, X_prismatic_Y,
                           RigidBodyInertia::from_solid_box(m_u, Vec3(0.15, 0.15, 0.15)), "wheel");
        sys.model.add_body(0, prismatic, X_prismatic_Y, X_prismatic_Y,
                           RigidBodyInertia::from_solid_box(m_s, Vec3(0.5, 0.2, 0.4)), "chassis");
    }

    /// The quarter car with its suspension spring-damper and tyre.
    mbd::kernel::System make_quarter_car(mbd::Real susp_damping)
    {
        using namespace mbd;
        kernel::System sys;
        add_quarter_car_bodies(sys);

        // Suspension spring-damper between wheel and chassis origins
        sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
            1, 2, Vec3::Zero(), Vec3::Zero(), k_s, susp_damping, L0_s));

        // Tire contact force on wheel, no tire damping for clean frequency tests
        sys.force_elements.push_back(std::make_shared<TireContactForce>(1, R_free, k_t, 0.0));
        return sys;
    }

    void set_heights(mbd::kernel::Simulator& sim, mbd::Real y_wheel, mbd::Real y_chassis)
    {
        sim.q(0) = y_wheel;
        sim.q(1) = y_chassis;
        sim.v.setZero();
        sim.initialize();
    }

    struct Sample {
        mbd::Real time, y_chassis, v_chassis;
    };

    /// Chassis height and velocity at the start and after every step.
    std::vector<Sample> record(mbd::kernel::Simulator& sim, mbd::Real duration, mbd::Real dt)
    {
        std::vector<Sample> samples{{sim.time, sim.q(1), sim.v(1)}};
        const int steps = static_cast<int>(std::round(duration / dt));
        for (int k = 0; k < steps; ++k) {
            sim.step(dt);
            samples.push_back({sim.time, sim.q(1), sim.v(1)});
        }
        return samples;
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
    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    // Start slightly above equilibrium
    set_heights(sim, y_w_eq() + 0.02, y_c_eq() + 0.05);

    // Simulate long enough for damping to settle (5 seconds)
    sim.run(5.0, 0.001);

    // Should settle near analytical equilibrium
    REQUIRE_THAT(sim.q(0), WithinAbs(y_w_eq(), 0.002));
    REQUIRE_THAT(sim.q(1), WithinAbs(y_c_eq(), 0.002));

    // Velocities should be near zero
    REQUIRE_THAT(sim.v(0), WithinAbs(0.0, 0.01));
    REQUIRE_THAT(sim.v(1), WithinAbs(0.0, 0.01));
}

// ============================================================================
// Natural frequencies (undamped)
// ============================================================================

TEST_CASE("Quarter-car: undamped natural frequencies match analytical",
          "[quarter_car][frequency]")
{
    using namespace mbd;

    auto sys = make_quarter_car(0.0); // No damping
    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    // Start at equilibrium with a small chassis perturbation (excites body bounce mode)
    set_heights(sim, y_w_eq(), y_c_eq() + 0.01);  // 10mm bump

    // Simulate 3 seconds
    const auto history = record(sim, 3.0, 0.001);

    // Only the body bounce mode is measured here (~1.4 Hz); the wheel-hop
    // frequency (the second element, ~11.8 Hz) is not.
    const Real omega1 = analytical_frequencies().first;
    const Real f1_theory = omega1 / (2.0 * pi);

    // Measure body bounce frequency from chassis displacement history.
    // Count zero crossings of (y_c - y_c_eq).
    const Real y_c_0 = y_c_eq();
    std::vector<Real> cross_times;
    for (size_t k = 1; k < history.size(); ++k) {
        const Real dy_prev = history[k - 1].y_chassis - y_c_0;
        const Real dy_curr = history[k].y_chassis - y_c_0;

        // Negative-going crossing
        if (dy_prev > 0.0 && dy_curr <= 0.0) {
            const Real t0 = history[k - 1].time;
            const Real t1 = history[k].time;
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
    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    // Start at equilibrium with perturbation
    set_heights(sim, y_w_eq() + 0.005, y_c_eq() + 0.01);

    auto compute_energy = [&]() -> Real {
        // Kinetic and gravitational energy of the bodies
        const Real KE = kernel::kinetic_energy(sys.model, sim.data());
        const Real PE_grav = kernel::potential_energy(sys.model, sim.data());

        // Suspension spring PE: 0.5 * k_s * (dist - L0)^2
        const Real dist_susp = (sim.states()[2].p_WB - sim.states()[1].p_WB).norm();
        const Real PE_susp = 0.5 * k_s * (dist_susp - L0_s) * (dist_susp - L0_s);

        // Tire spring PE: 0.5 * k_t * deflection^2
        const auto* tire = static_cast<const TireContactForce*>(sys.force_elements[1].get());
        const Real defl = tire->get_deflection(sim.states());
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
    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    set_heights(sim, y_w_eq(), y_c_eq() + 0.03);  // 30mm perturbation

    const auto history = record(sim, 3.0, 0.001);

    // Measure peak chassis displacement over time
    const Real y_eq = y_c_eq();
    Real max_disp_first_half = 0.0;
    Real max_disp_second_half = 0.0;

    for (const auto& rec : history) {
        const Real disp = std::abs(rec.y_chassis - y_eq);
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
    for (const auto& rec : history) {
        max_vel = std::max(max_vel, std::abs(rec.v_chassis));
    }
    const Real final_vel = std::abs(sim.v(1));
    REQUIRE(final_vel < max_vel * 0.1);
}

// ============================================================================
// Force element projection consistency
// ============================================================================

TEST_CASE("Quarter-car: force projection gives same result as manual force callback",
          "[quarter_car][projection]")
{
    using namespace mbd;

    // System 1: the force elements, turned into generalized forces by the kernel
    auto sys1 = make_quarter_car(c_s);
    kernel::Simulator sim1(sys1);
    sim1.method = kernel::Integrator::RK4;
    sim1.q(0) = y_w_eq() + 0.01;
    sim1.q(1) = y_c_eq() + 0.02;
    sim1.v(0) = 0.1;
    sim1.v(1) = -0.05;
    sim1.initialize();
    const VecX qdd1 = sim1.acceleration(sim1.q, sim1.v, 0.0);

    // The generalized forces of the same body forces, computed separately
    kernel::Data data(sys1.model);
    kernel::forward_kinematics(sys1.model, data, sim1.q, sim1.v);
    VecX tau_projected;
    kernel::generalized_forces(sys1.model, data, sim1.forces(), tau_projected);

    // System 2: no force elements; the same generalized forces applied directly
    kernel::System sys2;
    add_quarter_car_bodies(sys2);
    kernel::Simulator sim2(sys2);
    sim2.q = sim1.q;
    sim2.v = sim1.v;
    sim2.tau = tau_projected;
    const VecX qdd2 = sim2.acceleration(sim2.q, sim2.v, 0.0);

    REQUIRE_THAT(qdd1(0), WithinAbs(qdd2(0), 1e-10));
    REQUIRE_THAT(qdd1(1), WithinAbs(qdd2(1), 1e-10));

    // For two vertical sliders, the generalized forces are the vertical body forces
    REQUIRE_THAT(tau_projected(0), WithinAbs(sim1.forces()[1].f_W.y(), 1e-9));
    REQUIRE_THAT(tau_projected(1), WithinAbs(sim1.forces()[2].f_W.y(), 1e-9));
}

// ============================================================================
// Tire lifts off
// ============================================================================

TEST_CASE("Quarter-car: tire lifts off when wheel is above free radius",
          "[quarter_car][liftoff]")
{
    using namespace mbd;

    kernel::System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const Transform3 X_prismatic_Y = Transform3::FromRotation(
        Mat3(Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix()));
    sys.model.add_body(0, std::make_shared<kernel::PrismaticJointModel>(), X_prismatic_Y,
                       X_prismatic_Y, RigidBodyInertia::from_solid_box(m_u, Vec3(0.15, 0.15, 0.15)),
                       "wheel");

    auto tire = std::make_shared<TireContactForce>(1, R_free, k_t, 0.0);
    const TireContactForce* tire_ptr = tire.get();
    sys.force_elements.push_back(std::move(tire));

    kernel::Simulator sim(sys);
    auto apply_at = [&](Real y_wheel) {
        sim.q(0) = y_wheel;
        sim.v.setZero();
        sim.acceleration(sim.q, sim.v, 0.0);   // states and forces at this height
    };

    // Wheel at exactly free radius: contact point at y=0, no penetration
    apply_at(R_free);
    REQUIRE_THAT(tire_ptr->get_vertical_force(sim.states()), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(sim.forces()[1].f_W.norm(), WithinAbs(0.0, 1e-9));

    // Wheel above free radius: no contact
    apply_at(R_free + 0.1);
    REQUIRE_THAT(tire_ptr->get_vertical_force(sim.states()), WithinAbs(0.0, 1e-9));

    // Wheel below free radius: contact force
    apply_at(R_free - 0.01);  // 10mm compression
    const Real expected_force = k_t * 0.01; // 200000 * 0.01 = 2000 N
    REQUIRE_THAT(tire_ptr->get_vertical_force(sim.states()), WithinAbs(expected_force, 1.0));
    REQUIRE(sim.forces()[1].f_W.y() > 0.0); // Pushes up
}
