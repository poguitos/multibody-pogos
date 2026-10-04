#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "mbd/forces/aerodynamics.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

// Aerodynamic forces. The force element reads only the chassis's world
// state, so most cases give it one directly; the rest simulate on the kernel
// (plan task 2.7).

using Catch::Matchers::WithinAbs;

namespace
{
    using mbd::Real;
    using mbd::Vec3;

    /// The force of an aero element on a chassis (body 1) at position p,
    /// moving at velocity v, not rotating.
    mbd::RigidBodyForces aero_force(const mbd::AerodynamicForce& aero, const Vec3& p, const Vec3& v)
    {
        std::vector<mbd::RigidBodyState> states(2);
        states[1] = mbd::RigidBodyState(p, mbd::Quat::Identity(), v, Vec3::Zero());
        std::vector<mbd::RigidBodyForces> forces(2);
        aero.apply(states, forces);
        return forces[1];
    }

    Real forward_speed(const mbd::kernel::Simulator& sim, mbd::BodyIndex chassis)
    {
        const auto& s = sim.states()[static_cast<std::size_t>(chassis)];
        return s.v_WB.dot(s.q_WB * Vec3::UnitX());
    }
}

// ============================================================================
// Static drag force
// ============================================================================

TEST_CASE("Aero: zero velocity produces zero force", "[aero][static]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 1.5;
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3::Zero());   // at rest

    REQUIRE_THAT(f.f_W.norm(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(f.tau_W.norm(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Aero: drag force opposes horizontal velocity", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 0.0;
    AerodynamicForce aero(1, p);

    // Chassis moving forward at 30 m/s
    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(30.0, 0.0, 0.0));

    // Drag = 0.5 * 1.225 * 30^2 * 0.7 = 385.875 N, opposing +X
    const Real expected_drag = 0.5 * 1.225 * 30.0 * 30.0 * 0.7;
    REQUIRE_THAT(f.f_W.x(), WithinAbs(-expected_drag, 1.0));
    REQUIRE_THAT(f.f_W.y(), WithinAbs(0.0, 1.0));
    REQUIRE_THAT(f.f_W.z(), WithinAbs(0.0, 1.0));
}

TEST_CASE("Aero: drag scales as V^2", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(1, p);

    auto get_drag_at_speed = [&](Real V) -> Real {
        return -aero_force(aero, Vec3::Zero(), Vec3(V, 0.0, 0.0)).f_W.x();
    };

    Real F_10 = get_drag_at_speed(10.0);
    Real F_20 = get_drag_at_speed(20.0);
    Real F_40 = get_drag_at_speed(40.0);

    // F scales with V^2: F_20 should be 4x F_10, F_40 should be 16x F_10
    REQUIRE_THAT(F_20 / F_10, WithinAbs(4.0, 0.01));
    REQUIRE_THAT(F_40 / F_10, WithinAbs(16.0, 0.01));
}

TEST_CASE("Aero: downforce acts in -Y direction", "[aero][lift]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 0.0;
    p.ClA = 2.0;
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(50.0, 0.0, 0.0));  // 180 km/h

    // Downforce = 0.5 * 1.225 * 50^2 * 2.0 = 3062.5 N
    const Real expected_DF = 0.5 * 1.225 * 50.0 * 50.0 * 2.0;
    REQUIRE_THAT(f.f_W.y(), WithinAbs(-expected_DF, 5.0));

    // No drag (CdA = 0)
    REQUIRE_THAT(f.f_W.x(), WithinAbs(0.0, 5.0));
}

TEST_CASE("Aero: drag opposes lateral velocity too", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(1, p);

    // Forward 20 m/s, lateral 15 m/s
    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(20.0, 0.0, 15.0));

    // Drag opposes (20, 0, 15), magnitude = sqrt(400+225)*scale
    const Real V = std::sqrt(20.0*20.0 + 15.0*15.0);
    const Real F_mag = 0.5 * 1.225 * V * V * 1.0;
    const Vec3 v_unit(20.0 / V, 0.0, 15.0 / V);
    const Vec3 expected_F = -F_mag * v_unit;

    REQUIRE_THAT(f.f_W.x(), WithinAbs(expected_F.x(), 1.0));
    REQUIRE_THAT(f.f_W.z(), WithinAbs(expected_F.z(), 1.0));
}

TEST_CASE("Aero: vertical velocity doesn't affect drag", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(1, p);

    // Forward 30 m/s, vertical 10 m/s (jumping or falling)
    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(30.0, 10.0, 0.0));

    // Drag should be based on horizontal V = 30, not on full V
    const Real expected_drag = 0.5 * 1.225 * 30.0 * 30.0 * 1.0;
    REQUIRE_THAT(f.f_W.x(), WithinAbs(-expected_drag, 1.0));
    REQUIRE_THAT(f.f_W.y(), WithinAbs(0.0, 1.0));
}

// ============================================================================
// Center of pressure produces moment
// ============================================================================

TEST_CASE("Aero: CoP above CG produces a pitch moment from drag",
          "[aero][cop]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    p.cop_offset_chassis = Vec3(0.0, 0.5, 0.0);  // CoP above CG (e.g., wing height)
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(30.0, 0.0, 0.0));

    // Drag F acts in -X at the CoP, 0.5 m above the CG. Its moment about the
    // CG is r x F = (0, 0.5, 0) x (-F, 0, 0) = (0, 0, 0.5 F): about Z, the
    // lateral axis of this Y-up layer, so a pitch moment. A positive rotation
    // about Z turns +X toward +Y: drag high above the CG lifts the nose.
    const Real drag = 0.5 * 1.225 * 30.0 * 30.0 * 1.0;
    REQUIRE_THAT(f.tau_W.z(), WithinAbs(0.5 * drag, 1e-9));
    REQUIRE_THAT(f.tau_W.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(f.tau_W.y(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// Ride-height-dependent downforce
// ============================================================================

TEST_CASE("Aero: ride-height sensitivity increases downforce at low h",
          "[aero][groundeffect]")
{
    using namespace mbd;

    AeroParams p;
    p.ClA = 1.0;
    p.h_ref = 0.10;
    p.dClA_dh = 5.0;  // strong ground effect
    AerodynamicForce aero(1, p);

    auto get_DF_at_h = [&](Real h, Real V) -> Real {
        return -aero_force(aero, Vec3(0.0, h, 0.0), Vec3(V, 0.0, 0.0)).f_W.y();
    };

    // At h = h_ref = 0.10: ClA_eff = 1.0 (baseline)
    // At h = 0.05 (below ref): ClA_eff = 1.0 + 5.0 * (0.10 - 0.05) = 1.25
    // At h = 0.15 (above ref): ClA_eff = 1.0 (no boost when above)

    Real DF_at_ref = get_DF_at_h(0.10, 50.0);
    Real DF_low    = get_DF_at_h(0.05, 50.0);
    Real DF_high   = get_DF_at_h(0.15, 50.0);

    // Low ride height: more downforce
    REQUIRE(DF_low > DF_at_ref);

    // High ride height: same as baseline
    REQUIRE_THAT(DF_high, WithinAbs(DF_at_ref, 1.0));

    // The increase should match: ClA goes from 1.0 to 1.25, 25% more
    const Real expected_ratio = 1.25 / 1.0;
    REQUIRE_THAT(DF_low / DF_at_ref, WithinAbs(expected_ratio, 0.01));
}

// ============================================================================
// Terminal velocity in free flight
// ============================================================================

TEST_CASE("Aero: terminal velocity from constant forward force", "[aero][terminal]")
{
    using namespace mbd;

    kernel::System sys;
    sys.model.gravity = Vec3::Zero();   // no gravity, to isolate aero
    const BodyIndex chassis = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
        Transform3::Identity(), RigidBodyInertia::from_solid_box(1500.0, Vec3(1.5, 0.3, 0.8)),
        "chassis");

    AeroParams p;
    p.CdA = 0.7;
    sys.force_elements.push_back(std::make_shared<AerodynamicForce>(chassis, p));

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    sim.initialize();

    // Constant forward force = 5000 N, along the chassis's X axis
    const Real F_const = 5000.0;
    sim.force_callback = [&](kernel::Simulator&, Real, VecX& tau) {
        tau(3) += F_const;
    };

    sim.run(180.0, 0.01);  // 3 minutes — drag rises slowly to terminal

    // Terminal velocity: F_drag = F_const → 0.5*rho*V^2*CdA = F_const
    // V_term = sqrt(2*F_const / (rho*CdA))
    const Real V_term_expected = std::sqrt(2.0 * F_const / (1.225 * 0.7));

    // Should be close to terminal (within 5%, since approach is asymptotic)
    REQUIRE_THAT(forward_speed(sim, chassis), WithinAbs(V_term_expected, V_term_expected * 0.05));
}

// ============================================================================
// Vehicle template integration
// ============================================================================

TEST_CASE("Aero: install_aerodynamics returns nullptr when both CdA and ClA are zero",
          "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.0;
    tmpl.chassis.ClA = 0.0;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero == nullptr);
}

TEST_CASE("Aero: install_aerodynamics adds force element", "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.7;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    const size_t n_before = sys.force_elements.size();
    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero != nullptr);
    REQUIRE(sys.force_elements.size() == n_before + 1);
}

TEST_CASE("Aero: drivetrain + aero coexist without exceptions",
          "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.7;
    tmpl.chassis.ClA = 0.0;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();

    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero != nullptr);

    // Let the vehicle settle at idle (no throttle) for half a second first.
    // This avoids transient mismatches between drivetrain initial state and
    // chassis state.
    dt.throttle = 0.0;
    REQUIRE_NOTHROW(sim.run(0.5, 0.001));

    // Now apply moderate throttle and run.
    dt.throttle = 0.3;
    REQUIRE_NOTHROW(sim.run(2.0, 0.001));

    // Vehicle should be moving forward, at a sane speed.
    REQUIRE(forward_speed(sim, vh.chassis_body) > 0.5);
    REQUIRE(forward_speed(sim, vh.chassis_body) < 80.0);
}

TEST_CASE("Aero: downforce reduces ride height at speed", "[aero][downforce]")
{
    using namespace mbd;

    // Run two simulations: one without aero, one with aero. Compare ride heights
    // at the same simulation time, immediately after a short settle.
    auto run_scenario = [](Real ClA) -> Real {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.chassis.CdA = 0.0;
        tmpl.chassis.ClA = ClA;

        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);

        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);

        // Set the speed gently — give the system stable forward motion
        const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;
        sim.v(v_forward) = 30.0;
        sim.initialize();

        if (ClA > 0.0) {
            vh.install_aerodynamics(sys);
        }

        // Light speed-control to keep velocity ~constant at 30 m/s
        sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
            tau(v_forward) += 200.0 * (30.0 - forward_speed(s, vh.chassis_body));
        };

        // Short run to allow vertical equilibration
        sim.run(0.5, 0.0005);

        return sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.y();
    };

    const Real h_no_aero = run_scenario(0.0);
    const Real h_with_aero = run_scenario(3.0);

    INFO("h_no_aero = " << h_no_aero << ", h_with_aero = " << h_with_aero);

    // With downforce, chassis should be lower (smaller Y) at the same speed
    REQUIRE(h_with_aero < h_no_aero);

    const Real dh = h_no_aero - h_with_aero;
    REQUIRE(dh > 0.001);
    REQUIRE(dh < 0.5);
}
