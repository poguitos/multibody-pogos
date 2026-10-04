#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle/drivetrain.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// The drivetrain on the simple vehicle, simulated on the kernel (plan task 2.7).

namespace
{
    /// World X velocity of the chassis (body 1): the speed of a car driving along X.
    mbd::Real vx(const mbd::kernel::Simulator& sim)
    {
        return sim.states()[1].v_WB.x();
    }
}

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

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sim, vm);
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
    const Real Vx = vx(sim);
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

TEST_CASE("Drivetrain: braking deceleration matches the hand calculation",
          "[drivetrain][braking]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 20.0; // Start at 20 m/s
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.params.engine.inertia = 0.0; // keep the engine out of the inertia that is braked
    dt.initialize(sim, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    // Let it settle at speed for 0.3s
    sim.run(0.3, 0.001);

    // Half pedal: 3000 Nm over the four wheels, 65 % of it at the front.
    // That asks each front tyre for about 2.9 kN on 4.6 kN of load and each
    // rear tyre for 1.5 kN on 3.0 kN (friction used: 0.62 and 0.51), well
    // inside the tyre's peak friction of 1.1. No wheel is near its limit.
    dt.brake = 0.5;
    sim.run(0.5, 0.001);               // pitch and tyre deflection settle
    const Real V1 = vx(sim);
    sim.run(1.0, 0.001);
    const Real V2 = vx(sim);
    const Real decel = (V1 - V2) / 1.0;

    // Hand calculation, for wheels that roll without locking:
    //   wheel c:  I * (a / R_c) = T_c - F_c * R_c     (it slows with the car)
    //   car:      m * a = sum(F_c)
    //   =>        a = sum(T_c / R_c) / (m + sum(I / R_c^2))
    // with T_c the brake torque, F_c the braking force of the tyre, R_c the
    // rolling radius and I the spin inertia of one wheel.
    Real force_from_torque = 0.0;
    Real inertia_as_mass   = 0.0;
    for (int c = 0; c < 4; ++c) {
        const Real R = vm.tires[c]->get_rolling_radius();
        force_from_torque += dt.brake_torque_out[c] / R;
        inertia_as_mass   += dt.wheel_inertia[c] / (R * R);
    }
    const Real a_expected = force_from_torque / (vp.total_mass() + inertia_as_mass);

    INFO("measured " << decel << " m/s^2, expected " << a_expected << " m/s^2");
    REQUIRE_THAT(decel, WithinRel(a_expected, 0.02));

    // Brake torque per wheel: bias times the total, split left/right.
    REQUIRE_THAT(dt.brake_torque_out[0], WithinAbs(0.5 * 6000.0 * 0.65 / 2.0, 1e-9));
    REQUIRE_THAT(dt.brake_torque_out[2], WithinAbs(0.5 * 6000.0 * 0.35 / 2.0, 1e-9));

    // Every tyre brakes, driven or not, with the force its brake asks for
    // (less the small part that slows the wheel itself).
    for (int c = 0; c < 4; ++c) {
        const Real R = vm.tires[c]->get_rolling_radius();
        const Real F_expected =
            (dt.brake_torque_out[c] - dt.wheel_inertia[c] * a_expected / R) / R;
        INFO("corner " << c);
        REQUIRE_THAT(-vm.tires[c]->get_Fx(), WithinRel(F_expected, 0.03));
        REQUIRE(dt.wheel_omega[c] > 0.0); // still rolling
    }
}

TEST_CASE("Drivetrain: full braking decelerates at the grip limit",
          "[drivetrain][braking]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 20.0;
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sim, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    sim.run(0.3, 0.001);
    const Real V_before = vx(sim);

    // Full pedal asks for 6000 Nm, about 17.6 kN at the contact patches. The
    // tyres can transmit at most 1.1 * m * g = 16.8 kN, so the brakes are no
    // longer the limit: the deceleration is set by the tyres. It cannot
    // exceed peak friction, and it cannot fall below the friction of a locked,
    // sliding tyre (slip ratio -1).
    dt.brake = 1.0;
    const Real t_brake = 1.0;
    sim.run(t_brake, 0.001);
    const Real V_after = vx(sim);
    const Real decel = (V_before - V_after) / t_brake;

    const PacejkaTire tyre(vp.tire_params);
    const Real Fz_static = vp.weight_per_wheel();
    const Real mu_peak   = tyre.peak_mu_longitudinal(Fz_static);
    const Real mu_slide  = -tyre.compute(-1.0, 0.0, Fz_static).Fx / Fz_static;

    INFO("decel " << decel << " m/s^2, sliding mu " << mu_slide << ", peak mu " << mu_peak);
    // 10 % margin below sliding friction: it falls slightly on the unloaded
    // rear tyres, and the first tenths of a second are spent building force.
    REQUIRE(decel > 0.90 * mu_slide * g_accel);
    REQUIRE(decel < mu_peak * g_accel);

    // All four tyres brake; no wheel turns backwards.
    for (int c = 0; c < 4; ++c) {
        INFO("corner " << c);
        REQUIRE(vm.tires[c]->get_Fx() < 0.0);
        REQUIRE(dt.wheel_omega[c] >= 0.0);
    }
}

// ============================================================================
// Coasting (no throttle, no brake) maintains speed approximately
// ============================================================================

TEST_CASE("Drivetrain: coasting approximately maintains speed", "[drivetrain][coast]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 15.0;
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sim, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    sim.run(0.3, 0.001); // settle
    const Real V_start = vx(sim);

    sim.run(2.0, 0.001);
    const Real V_end = vx(sim);

    // Nothing in this model takes energy out of a coasting car: there is no
    // aerodynamic drag, no rolling resistance and no engine braking. The
    // speed must stay where it is.
    const Real speed_loss_fraction = (V_start - V_end) / V_start;
    REQUIRE(std::abs(speed_loss_fraction) < 1e-3);
}

// ============================================================================
// Auto-shift logic
// ============================================================================

TEST_CASE("Drivetrain: auto-shift changes up at the shift speed of each gear",
          "[drivetrain][shift]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sim, vm);
    dt.connect(sim, vm);

    const auto& gp = dt.params.gearbox;

    // Wheel speed at which the engine reaches the up-shift speed in gear g.
    auto shift_speed = [&](int gear) {
        return gp.shift_up_rpm * 2.0 * pi / 60.0 / (gp.ratios[gear - 1] * gp.final_drive);
    };

    // Accelerate with full throttle and watch every gear change.
    dt.throttle = 1.0;
    Real omega_max = 0.0;
    int upshifts = 0;
    for (int i = 0; i < 10500; ++i) {
        const int gear_before = dt.current_gear;
        const Real omega_before = 0.5 * (dt.wheel_omega[2] + dt.wheel_omega[3]);
        sim.step(0.001);
        omega_max = std::max(omega_max, omega_before);

        if (dt.current_gear != gear_before) {
            // Only upward, one gear at a time, and only once the driven wheels
            // have passed the shift speed of the gear that is left.
            REQUIRE(dt.current_gear == gear_before + 1);
            REQUIRE(omega_before > shift_speed(gear_before));
            ++upshifts;
        }
    }

    // The gear is the number of shift speeds the driven wheels have passed.
    int expected_gear = 1;
    while (expected_gear < dt.num_gears() && omega_max > shift_speed(expected_gear)) {
        ++expected_gear;
    }
    REQUIRE(dt.current_gear == expected_gear);
    REQUIRE(upshifts == expected_gear - 1);

    // Lower bound on how far it must have got. Up to the change into third the
    // car accelerates at 2.88 m/s^2 or more:
    //   - engine-limited at the bottom of first gear: 400 Nm * 0.4 (idle
    //     fraction) * 12.25 * 0.92 / 0.34 m = 5.3 kN on 1840 kg (car, wheels
    //     and the engine inertia seen through first gear) = 2.88 m/s^2;
    //   - with the driven wheels spinning: sliding friction 0.65 on at least
    //     the static rear axle load of 7.65 kN = 4.9 kN, or 3.0 m/s^2;
    //   - at the top of second gear: 6.6 kN on 1730 kg = 3.8 m/s^2.
    // Third gear is taken at 84.6 rad/s, about 28.8 m/s, so no later than
    // 28.8 / 2.88 = 10.0 s after the start.
    REQUIRE(dt.current_gear >= 3);
    REQUIRE(vx(sim) > 2.88 * 10.0);

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

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::FWD;
    dt.initialize(sim, vm);
    dt.throttle = 1.0;
    dt.connect(sim, vm);

    sim.run(3.0, 0.001);

    // Vehicle should be moving
    REQUIRE(vx(sim) > 3.0);

    // The rear wheels are not driven: they roll freely at the speed of the
    // car. The front wheels pull the car, so they turn slightly faster than
    // free rolling (positive slip).
    const Real omega_front_avg = 0.5 * (dt.wheel_omega[0] + dt.wheel_omega[1]);
    const Real omega_rear_avg  = 0.5 * (dt.wheel_omega[2] + dt.wheel_omega[3]);
    const Real R_rear = vm.tires[2]->get_rolling_radius();
    REQUIRE_THAT(omega_rear_avg * R_rear, WithinRel(vx(sim), 0.01));
    REQUIRE(omega_front_avg > omega_rear_avg);
}

// ============================================================================
// Drivetrain with steering
// ============================================================================

TEST_CASE("Drivetrain: RWD vehicle corners under power", "[drivetrain][cornering]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 10.0;
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sim, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    // Maintain speed via external force callback (gentle controller)
    const Real V_target = 10.0;
    const Real K_speed = 500.0;
    sim.force_callback = [&](kernel::Simulator& s, Real /*t*/, VecX& tau) {
        const Vec3 fwd_W = s.states()[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states()[vm.chassis_body].v_WB.dot(fwd_W);
        tau(3) += K_speed * (V_target - Vx);
    };

    // Settle at speed
    sim.run(0.5, 0.001);

    // Apply steering
    vm.set_front_steering(0.02);
    const Real z_before = sim.states()[vm.chassis_body].p_WB.z();

    sim.run(3.0, 0.001);
    const Real z_after = sim.states()[vm.chassis_body].p_WB.z();

    // Vehicle should turn left (positive Z)
    REQUIRE(z_after - z_before > 0.05);

    // Vehicle should still be on the ground
    REQUIRE_THAT(sim.states()[vm.chassis_body].p_WB.y(),
                 WithinAbs(vp.chassis_height_eq(), 0.05));
}

// ============================================================================
// Initialization sets correct wheel omegas
// ============================================================================

TEST_CASE("Drivetrain: initialize matches wheel omega to vehicle speed",
          "[drivetrain][init]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams vp;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 20.0;
    sim.initialize();

    Drivetrain dt;
    dt.initialize(sim, vm);

    const Real R_eff = vp.tire_free_radius * 0.97;
    const Real expected_omega = 20.0 / R_eff;

    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(dt.wheel_omega[c], WithinAbs(expected_omega, 0.1));
    }
}