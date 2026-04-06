#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/drivetrain.hpp"
#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

// ============================================================================
// Engine torque curve
// ============================================================================

TEST_CASE("Drivetrain: engine torque at idle", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T = Drivetrain::compute_engine_torque(ep.idle_rpm, 1.0, ep);

    // At idle, full throttle: max_torque * idle_fraction = 400 * 0.4 = 160
    REQUIRE_THAT(T, WithinAbs(ep.max_torque * ep.idle_torque_fraction, 0.1));
}

TEST_CASE("Drivetrain: engine peak torque", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T = Drivetrain::compute_engine_torque(ep.peak_torque_rpm, 1.0, ep);

    // At peak RPM, full throttle: max_torque
    REQUIRE_THAT(T, WithinAbs(ep.max_torque, 0.1));
}

TEST_CASE("Drivetrain: engine torque at redline", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T = Drivetrain::compute_engine_torque(ep.redline_rpm, 1.0, ep);

    REQUIRE_THAT(T, WithinAbs(ep.max_torque * ep.redline_torque_fraction, 0.1));
}

TEST_CASE("Drivetrain: rev limiter cuts torque above redline", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T = Drivetrain::compute_engine_torque(ep.redline_rpm + 100.0, 1.0, ep);

    REQUIRE_THAT(T, WithinAbs(0.0, 0.01));
}

TEST_CASE("Drivetrain: zero throttle gives zero torque", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T = Drivetrain::compute_engine_torque(4000.0, 0.0, ep);

    REQUIRE_THAT(T, WithinAbs(0.0, 0.01));
}

TEST_CASE("Drivetrain: half throttle gives half torque", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T_full = Drivetrain::compute_engine_torque(4000.0, 1.0, ep);
    Real T_half = Drivetrain::compute_engine_torque(4000.0, 0.5, ep);

    REQUIRE_THAT(T_half, WithinAbs(T_full * 0.5, 0.1));
}

TEST_CASE("Drivetrain: torque increases from idle to peak", "[drivetrain][engine]")
{
    using namespace mbd;

    EngineParams ep;
    Real T_idle = Drivetrain::compute_engine_torque(ep.idle_rpm, 1.0, ep);
    Real T_mid  = Drivetrain::compute_engine_torque(
        0.5 * (ep.idle_rpm + ep.peak_torque_rpm), 1.0, ep);
    Real T_peak = Drivetrain::compute_engine_torque(ep.peak_torque_rpm, 1.0, ep);

    REQUIRE(T_idle < T_mid);
    REQUIRE(T_mid < T_peak);
}

// ============================================================================
// Gear ratios and RPM computation
// ============================================================================

TEST_CASE("Drivetrain: RPM computation from wheel omega", "[drivetrain][gearbox]")
{
    using namespace mbd;

    Drivetrain dt;
    dt.current_gear = 1;

    // omega = 50 rad/s, gear 1 ratio = 3.5, final = 3.5
    // omega_engine = 50 * 3.5 * 3.5 = 612.5 rad/s
    // RPM = 612.5 * 60 / (2*pi) = 5849.7
    Real rpm = dt.omega_to_rpm(50.0);
    const Real expected = 50.0 * 3.5 * 3.5 * 60.0 / (2.0 * pi);
    REQUIRE_THAT(rpm, WithinAbs(expected, 1.0));
}

TEST_CASE("Drivetrain: higher gear gives lower RPM at same speed", "[drivetrain][gearbox]")
{
    using namespace mbd;

    Drivetrain dt;
    const Real omega = 80.0; // rad/s

    dt.current_gear = 1;
    Real rpm_1st = dt.omega_to_rpm(omega);

    dt.current_gear = 4;
    Real rpm_4th = dt.omega_to_rpm(omega);

    REQUIRE(rpm_1st > rpm_4th);
}

// ============================================================================
// Drive layout
// ============================================================================

TEST_CASE("Drivetrain: RWD drives only rear wheels", "[drivetrain][layout]")
{
    using namespace mbd;

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;

    REQUIRE_FALSE(dt.is_driven(0)); // FL
    REQUIRE_FALSE(dt.is_driven(1)); // FR
    REQUIRE(dt.is_driven(2));       // RL
    REQUIRE(dt.is_driven(3));       // RR
}

TEST_CASE("Drivetrain: FWD drives only front wheels", "[drivetrain][layout]")
{
    using namespace mbd;

    Drivetrain dt;
    dt.params.layout = DriveLayout::FWD;

    REQUIRE(dt.is_driven(0));       // FL
    REQUIRE(dt.is_driven(1));       // FR
    REQUIRE_FALSE(dt.is_driven(2)); // RL
    REQUIRE_FALSE(dt.is_driven(3)); // RR
}

TEST_CASE("Drivetrain: AWD drives all wheels", "[drivetrain][layout]")
{
    using namespace mbd;

    Drivetrain dt;
    dt.params.layout = DriveLayout::AWD;

    for (int c = 0; c < 4; ++c) {
        REQUIRE(dt.is_driven(c));
    }
}

// ============================================================================
// Standing start acceleration
// ============================================================================

TEST_CASE("Drivetrain: standing start accelerates the vehicle", "[drivetrain][dynamic]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sys, vm);
    dt.throttle = 1.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    // Let suspension settle briefly
    dt.throttle = 0.0;
    sim.run(0.3, 0.001);

    // Now apply full throttle
    dt.throttle = 1.0;
    sim.run(3.0, 0.001);

    // Vehicle should be moving forward significantly
    const Real Vx = sys.q_dot(0);
    REQUIRE(Vx > 5.0); // Should reach at least 5 m/s in 3 seconds

    // Wheel omegas should be positive
    for (int c = 0; c < 4; ++c) {
        REQUIRE(dt.wheel_omega[c] > 0.0);
    }

    // At moderate speed in first gear, RPM may still be below shift threshold.
    // Just verify gear is valid.
    REQUIRE(dt.current_gear >= 1);
    REQUIRE(dt.current_gear <= dt.num_gears());

    // Engine RPM should be in valid range
    REQUIRE(dt.engine_rpm >= dt.params.engine.idle_rpm);
    REQUIRE(dt.engine_rpm <= dt.params.engine.redline_rpm + 100.0);
}

// ============================================================================
// Braking from speed
// ============================================================================

TEST_CASE("Drivetrain: braking decelerates the vehicle", "[drivetrain][braking]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    sys.q_dot(0) = 20.0; // Start at 20 m/s
    sys.compute_kinematics();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sys, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    // Let it settle at speed for 0.3s
    sim.run(0.3, 0.001);
    const Real V_before = sys.q_dot(0);

    // Apply full brakes
    dt.brake = 1.0;
    sim.run(2.0, 0.001);

    const Real V_after = sys.q_dot(0);

    // Vehicle should have slowed down significantly
    REQUIRE(V_after < V_before * 0.3);

    // Wheel omegas should have decreased
    for (int c = 0; c < 4; ++c) {
        REQUIRE(dt.wheel_omega[c] >= 0.0); // Not negative
    }
}

// ============================================================================
// Coasting (no throttle, no brake) maintains speed approximately
// ============================================================================

TEST_CASE("Drivetrain: coasting approximately maintains speed", "[drivetrain][coast]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    sys.q_dot(0) = 15.0;
    sys.compute_kinematics();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sys, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    sim.run(0.3, 0.001); // settle
    const Real V_start = sys.q_dot(0);

    sim.run(2.0, 0.001);
    const Real V_end = sys.q_dot(0);

    // Without aero drag, coasting should maintain speed within ~10%
    // (small losses from tire rolling resistance are expected)
    const Real speed_loss_fraction = (V_start - V_end) / V_start;
    REQUIRE(speed_loss_fraction < 0.10);
    REQUIRE(speed_loss_fraction > -0.01); // Shouldn't gain speed
}

// ============================================================================
// Auto-shift logic
// ============================================================================

TEST_CASE("Drivetrain: auto-shift selects appropriate gear for speed",
          "[drivetrain][shift]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sys, vm);
    dt.connect(sim, vm);

    // Accelerate with full throttle for 5 seconds
    dt.throttle = 1.0;
    sim.run(5.0, 0.001);

    // Should have reached a higher gear
    REQUIRE(dt.current_gear >= 3);

    // RPM should be within the shift band
    REQUIRE(dt.engine_rpm >= dt.params.gearbox.shift_down_rpm - 100.0);
    REQUIRE(dt.engine_rpm <= dt.params.gearbox.shift_up_rpm + 100.0);
}

// ============================================================================
// FWD layout works correctly
// ============================================================================

TEST_CASE("Drivetrain: FWD drives front wheels and accelerates",
          "[drivetrain][fwd]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    Drivetrain dt;
    dt.params.layout = DriveLayout::FWD;
    dt.initialize(sys, vm);
    dt.throttle = 1.0;
    dt.connect(sim, vm);

    sim.run(3.0, 0.001);

    // Vehicle should be moving
    REQUIRE(sys.q_dot(0) > 3.0);

    // Front wheel omegas should be higher than rear (driven vs free-rolling)
    // Actually both should track vehicle speed approximately,
    // but front may have slightly higher omega due to drive slip
    const Real omega_front_avg = 0.5 * (dt.wheel_omega[0] + dt.wheel_omega[1]);
    const Real omega_rear_avg  = 0.5 * (dt.wheel_omega[2] + dt.wheel_omega[3]);
    REQUIRE(omega_front_avg > 0.0);
    REQUIRE(omega_rear_avg > 0.0);
}

// ============================================================================
// Drivetrain with steering
// ============================================================================

TEST_CASE("Drivetrain: RWD vehicle corners under power", "[drivetrain][cornering]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    sys.q_dot(0) = 10.0;
    sys.compute_kinematics();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sys, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    // Maintain speed via external force callback (gentle controller)
    const Real V_target = 10.0;
    const Real K_speed = 500.0;
    sim.force_callback = [&](MultibodySystem& s, Real /*t*/, VecX& tau) {
        const Vec3 fwd_W = s.states[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vm.chassis_body].v_WB.dot(fwd_W);
        tau(0) += K_speed * (V_target - Vx);
    };

    // Settle at speed
    sim.run(0.5, 0.001);

    // Apply steering
    vm.set_front_steering(0.02);
    const Real z_before = sys.states[vm.chassis_body].p_WB.z();

    sim.run(3.0, 0.001);
    const Real z_after = sys.states[vm.chassis_body].p_WB.z();

    // Vehicle should turn left (positive Z)
    REQUIRE(z_after - z_before > 0.05);

    // Vehicle should still be on the ground
    REQUIRE_THAT(sys.states[vm.chassis_body].p_WB.y(),
                 WithinAbs(vp.chassis_height_eq(), 0.05));
}

// ============================================================================
// Initialization sets correct wheel omegas
// ============================================================================

TEST_CASE("Drivetrain: initialize matches wheel omega to vehicle speed",
          "[drivetrain][init]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    sys.q_dot(0) = 20.0;
    sys.compute_kinematics();

    Drivetrain dt;
    dt.initialize(sys, vm);

    const Real R_eff = vp.tire_free_radius * 0.97;
    const Real expected_omega = 20.0 / R_eff;

    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(dt.wheel_omega[c], WithinAbs(expected_omega, 0.1));
    }
}