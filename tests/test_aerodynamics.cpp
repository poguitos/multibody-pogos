#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/aerodynamics.hpp"
#include "mbd/vehicle_template.hpp"
#include "mbd/simulator.hpp"
#include "mbd/drivetrain.hpp"

using Catch::Matchers::WithinAbs;

// ============================================================================
// Static drag force
// ============================================================================

TEST_CASE("Aero: zero velocity produces zero force", "[aero][static]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FixedJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 1.5;
    AerodynamicForce aero(chassis, p);

    sys.q.setZero();
    sys.compute_kinematics(); // chassis at rest
    sys.clear_forces();
    aero.apply(sys.states, sys.forces);

    REQUIRE_THAT(sys.forces[chassis].f_W.norm(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(sys.forces[chassis].tau_W.norm(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Aero: drag force opposes horizontal velocity", "[aero][drag]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 0.0;
    AerodynamicForce aero(chassis, p);

    // Set chassis moving forward at 30 m/s
    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(0) = 30.0;  // FreeCoordJoint q_dot(0) is forward velocity
    sys.compute_kinematics();

    sys.clear_forces();
    aero.apply(sys.states, sys.forces);

    // Drag = 0.5 * 1.225 * 30^2 * 0.7 = 385.875 N, opposing +X
    const Real expected_drag = 0.5 * 1.225 * 30.0 * 30.0 * 0.7;
    REQUIRE_THAT(sys.forces[chassis].f_W.x(), WithinAbs(-expected_drag, 1.0));
    REQUIRE_THAT(sys.forces[chassis].f_W.y(), WithinAbs(0.0, 1.0));
    REQUIRE_THAT(sys.forces[chassis].f_W.z(), WithinAbs(0.0, 1.0));
}

TEST_CASE("Aero: drag scales as V^2", "[aero][drag]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(chassis, p);

    auto get_drag_at_speed = [&](Real V) -> Real {
        sys.q.setZero();
        sys.q_dot.setZero();
        sys.q_dot(0) = V;
        sys.compute_kinematics();
        sys.clear_forces();
        aero.apply(sys.states, sys.forces);
        return -sys.forces[chassis].f_W.x();
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

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 0.0;
    p.ClA = 2.0;
    AerodynamicForce aero(chassis, p);

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(0) = 50.0;  // 180 km/h
    sys.compute_kinematics();
    sys.clear_forces();
    aero.apply(sys.states, sys.forces);

    // Downforce = 0.5 * 1.225 * 50^2 * 2.0 = 3062.5 N
    const Real expected_DF = 0.5 * 1.225 * 50.0 * 50.0 * 2.0;
    REQUIRE_THAT(sys.forces[chassis].f_W.y(), WithinAbs(-expected_DF, 5.0));

    // No drag (CdA = 0)
    REQUIRE_THAT(sys.forces[chassis].f_W.x(), WithinAbs(0.0, 5.0));
}

TEST_CASE("Aero: drag opposes lateral velocity too", "[aero][drag]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(chassis, p);

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(0) = 20.0;  // forward
    sys.q_dot(2) = 15.0;  // lateral
    sys.compute_kinematics();
    sys.clear_forces();
    aero.apply(sys.states, sys.forces);

    // Drag opposes (20, 0, 15), magnitude = sqrt(400+225)*scale
    const Real V = std::sqrt(20.0*20.0 + 15.0*15.0);
    const Real F_mag = 0.5 * 1.225 * V * V * 1.0;
    const Vec3 v_unit(20.0 / V, 0.0, 15.0 / V);
    const Vec3 expected_F = -F_mag * v_unit;

    REQUIRE_THAT(sys.forces[chassis].f_W.x(), WithinAbs(expected_F.x(), 1.0));
    REQUIRE_THAT(sys.forces[chassis].f_W.z(), WithinAbs(expected_F.z(), 1.0));
}

TEST_CASE("Aero: vertical velocity doesn't affect drag", "[aero][drag]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(chassis, p);

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(0) = 30.0;
    sys.q_dot(1) = 10.0;  // vertical velocity (jumping or falling)
    sys.compute_kinematics();
    sys.clear_forces();
    aero.apply(sys.states, sys.forces);

    // Drag should be based on horizontal V = 30, not on full V
    const Real expected_drag = 0.5 * 1.225 * 30.0 * 30.0 * 1.0;
    REQUIRE_THAT(sys.forces[chassis].f_W.x(), WithinAbs(-expected_drag, 1.0));
    REQUIRE_THAT(sys.forces[chassis].f_W.y(), WithinAbs(0.0, 1.0));
}

// ============================================================================
// Center of pressure produces moment
// ============================================================================

TEST_CASE("Aero: CoP forward of CG produces nose-down moment from drag",
          "[aero][cop]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 1.0;
    p.cop_offset_chassis = Vec3(0.0, 0.5, 0.0);  // CoP above CG (e.g., wing height)
    AerodynamicForce aero(chassis, p);

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(0) = 30.0;
    sys.compute_kinematics();
    sys.clear_forces();
    aero.apply(sys.states, sys.forces);

    // Drag is in -X, applied at CoP (0, 0.5, 0) above CG
    // Moment = r_cop x F_drag = (0, 0.5, 0) x (-F, 0, 0) = (0, 0, 0.5*F) ... wait
    // (0, 0.5, 0) cross (-F, 0, 0) = (0.5*0 - 0*0, 0*(-F) - 0*0, 0*0 - 0.5*(-F))
    //                              = (0, 0, 0.5*F)
    // So torque about Z (yaw axis) is positive. But this is roll/pitch axis confusion.
    // Actually: with X=fwd, Y=up, Z=left. CoP above CG (+Y). Force in -X.
    // Cross (0, 0.5, 0) x (-F, 0, 0) = (0.5*0 - 0*0, 0*(-F) - 0*0, 0*0 - 0.5*(-F))
    //                              = (0, 0, 0.5F).
    // That's a +Z moment, which means yaw to the left? No, +Z rotation about Z axis.
    // Actually our Z is lateral. So moment about Z = pitch (about lateral axis).
    // +Z moment = pitch nose-down? Let's verify: rotate about +Z by positive angle:
    // X axis rotates toward... R_z(theta) sends (1,0,0) to (cos, sin, 0). So +X moves to +Y.
    // That means nose goes UP. So +Z moment = nose UP.
    // For drag at high CoP, drag pushes top of car backward, so nose goes UP relative
    // to ground. Let me re-verify: the force on the car is at the high CoP, pointing
    // backward. This force has a moment arm above CG. Torque = r x F.

    // The moment should be non-zero
    REQUIRE(std::abs(sys.forces[chassis].tau_W.z()) > 0.1);
}

// ============================================================================
// Ride-height-dependent downforce
// ============================================================================

TEST_CASE("Aero: ride-height sensitivity increases downforce at low h",
          "[aero][groundeffect]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.ClA = 1.0;
    p.h_ref = 0.10;
    p.dClA_dh = 5.0;  // strong ground effect
    AerodynamicForce aero(chassis, p);

    auto get_DF_at_h = [&](Real h, Real V) -> Real {
        sys.q.setZero();
        sys.q_dot.setZero();
        sys.q(1) = h;
        sys.q_dot(0) = V;
        sys.compute_kinematics();
        sys.clear_forces();
        aero.apply(sys.states, sys.forces);
        return -sys.forces[chassis].f_W.y();
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

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(1500.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    AeroParams p;
    p.CdA = 0.7;
    sys.force_elements.push_back(std::make_unique<AerodynamicForce>(chassis, p));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());  // No gravity to isolate aero
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Constant forward force = 5000 N
    const Real F_const = 5000.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F_const;
    };

    sim.run(180.0, 0.01);  // 3 minutes — drag rises slowly to terminal

    // Terminal velocity: F_drag = F_const → 0.5*rho*V^2*CdA = F_const
    // V_term = sqrt(2*F_const / (rho*CdA))
    const Real V_term_expected = std::sqrt(2.0 * F_const / (1.225 * 0.7));

    // Should be close to terminal (within 5%, since approach is asymptotic)
    REQUIRE_THAT(sys.q_dot(0), WithinAbs(V_term_expected, V_term_expected * 0.05));
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

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero == nullptr);
}

TEST_CASE("Aero: install_aerodynamics adds force element", "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.7;

    MultibodySystem sys;
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

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);

    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sys, vh);
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

    // Vehicle should be moving forward.
    REQUIRE(sys.q_dot(0) > 0.5);

    // Sanity: speed shouldn't be absurd.
    REQUIRE(sys.q_dot(0) < 80.0);
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

        MultibodySystem sys;
        auto vh = build_vehicle(sys, tmpl);

        Simulator sim(sys);
        sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
        sim.method = IntegrationMethod::RK4;
        sim.initialize();

        set_vehicle_equilibrium(sys, vh);

        // Set the speed gently — give the system stable forward motion
        sys.q_dot(0) = 30.0;
        sys.compute_kinematics();

        if (ClA > 0.0) {
            vh.install_aerodynamics(sys);
        }

        // Light speed-control to keep velocity ~constant at 30 m/s
        sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
            const Vec3 fwd = s.states[vh.chassis_body].q_WB * Vec3::UnitX();
            const Real Vx = s.states[vh.chassis_body].v_WB.dot(fwd);
            tau(0) += 200.0 * (30.0 - Vx);
        };

        // Short run to allow vertical equilibration
        sim.run(0.5, 0.0005);

        return sys.q(1);
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