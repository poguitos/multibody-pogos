#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string>
#include <utility>

#include "mbd/vehicle/drivetrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// The drivetrain on the simple vehicle, simulated on the kernel (plan task 2.7);
// its wheels on spin joints since task 7.3 (D62).

namespace
{
    /// World X velocity of the chassis (body 1): the speed of a car driving along X.
    mbd::Real vx(const mbd::kernel::Simulator& sim)
    {
        return sim.states()[1].v_WB.x();
    }

    /// Start every wheel 3 % faster than free rolling, as the drivetrain
    /// started them before task 7.3 (its spins from R_free x 0.97): a release
    /// transient for the spins to settle from, which they no longer have when
    /// roll_wheels starts them at free rolling exactly.
    void spin_off_free_rolling(mbd::kernel::Simulator& sim, const mbd::VehicleModel& vm)
    {
        for (std::size_t c = 0; c < 4; ++c) sim.v(vm.spin_v[c]) *= 1.03;
        sim.refresh();
    }

    /// The simple car with its tyres as they were before task 7.4 in the two
    /// respects the implicit methods' tests below are about: without lag
    /// (the slips over max(|Vx|, V_low): at walking pace a stiff damper on
    /// each wheel's spin, the stiffness those methods are for) and without
    /// rolling resistance (nothing slows the coasting car). With the lag the
    /// spins couple to the tread's spring instead, and RK4 steps the car at
    /// 0.5 m/s in 5 ms (measured: every wheel within 1.3e-3 of free rolling).
    mbd::VehicleModel build_stiff_car(mbd::kernel::System& sys)
    {
        mbd::VehicleParams vp;
        vp.tyre.QSY1 = vp.tyre.QSY3 = vp.tyre.QSY4 = 0.0;
        mbd::VehicleModel vm = mbd::build_simple_vehicle(sys, vp);
        for (mbd::FullTireForce* t : vm.tires) t->lag = false;
        return vm;
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
    // The tyres' rolling resistance set aside (task 7.4; the coasting test
    // below has it): this is the brakes' torque against the car's and the
    // wheels' inertia.
    vp.tyre.QSY1 = vp.tyre.QSY3 = vp.tyre.QSY4 = 0.0;
    auto vm = build_simple_vehicle(sys, vp);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 20.0; // Start at 20 m/s
    sim.initialize();

    Drivetrain dt;
    dt.params.layout = DriveLayout::RWD;
    dt.initialize(sim, vm);
    dt.throttle = 0.0;
    dt.brake = 0.0;
    dt.connect(sim, vm);

    // Let it settle at speed for 0.3s
    sim.run(0.3, 0.001);

    // Half pedal: 3000 Nm over the four wheels, 65 % of it at the front.
    // That asks each front tyre for about 3.3 kN on 4.6 kN of load and each
    // rear tyre for 1.8 kN on 3.0 kN (friction used: 0.72 and 0.59), inside
    // the Appendix 3 tyre's peak friction of about 1.05. No wheel is near its
    // limit.
    dt.brake = 0.5;
    sim.run(0.5, 0.001);               // pitch and tyre deflection settle
    const Real V1 = vx(sim);
    sim.run(1.0, 0.001);
    const Real V2 = vx(sim);
    const Real decel = (V1 - V2) / 1.0;

    // Hand calculation, for wheels that roll without locking:
    //   wheel c:  I * (a / R_c) = T_c - F_c * r_c     (it slows with the car)
    //   car:      m * a = sum(F_c)
    //   =>        a = sum(T_c / r_c) / (m + sum(I / (R_c r_c)))
    // with T_c the brake torque, F_c the braking force of the tyre, R_c the
    // rolling radius (spin to speed), r_c the arm of F_c about the axle (task
    // 7.3: the force acts on the road, R_free - deflection below the axle),
    // and I the spin inertia of one wheel, m R^2 / 2 = 1.97 kg m^2. The slip
    // that the braking force needs (about -4 %) makes the wheels turn, and
    // slow, that much slower than rolling, so their inertia takes that much
    // less of the brakes' torque: of the wheels' 5 % share of the inertia,
    // 0.2 % of the deceleration, which this leaves out (measured 5.539
    // against 5.528 m/s^2). Allowed 0.5 %.
    Real force_from_torque = 0.0;
    Real inertia_as_mass   = 0.0;
    for (int c = 0; c < 4; ++c) {
        const Real R = vm.tires[c]->get_rolling_radius();
        const Real r = vm.tires[c]->R_free - vm.tires[c]->get_deflection();
        force_from_torque += dt.brake_torque_out[c] / r;
        inertia_as_mass   += dt.wheel_inertia[c] / (R * r);
    }
    const Real a_expected = force_from_torque / (vp.total_mass() + inertia_as_mass);

    INFO("measured " << decel << " m/s^2, expected " << a_expected << " m/s^2");
    REQUIRE_THAT(decel, WithinRel(a_expected, 0.005));

    // Brake torque per wheel: bias times the total, split left/right.
    REQUIRE_THAT(dt.brake_torque_out[0], WithinAbs(0.5 * 6000.0 * 0.65 / 2.0, 1e-9));
    REQUIRE_THAT(dt.brake_torque_out[2], WithinAbs(0.5 * 6000.0 * 0.35 / 2.0, 1e-9));

    // Every tyre brakes, driven or not, with the force its brake asks for
    // (less the small part that slows the wheel itself).
    for (int c = 0; c < 4; ++c) {
        const Real R = vm.tires[c]->get_rolling_radius();
        const Real r = vm.tires[c]->R_free - vm.tires[c]->get_deflection();
        const Real F_expected =
            (dt.brake_torque_out[c] - dt.wheel_inertia[c] * a_expected / R) / r;
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

    // Full pedal asks for 6000 Nm, about 19 kN at the contact patches (the
    // Appendix 3 tyre's radius, 0.31 m, task 7.4). The tyres can transmit
    // about 1.05 m g = 16 kN, so the brakes are no longer the limit: the
    // deceleration is set by the tyres. It cannot exceed the peak of the
    // braking curve at the static load, and it cannot fall below the friction
    // of a locked, sliding tyre (slip ratio -1). The peak bound holds with
    // the load transferred forward too: the peak friction falls with load
    // (pDx2 < 0, linearly), so the four tyres' peaks add up to most at equal
    // loads.
    dt.brake = 1.0;
    const Real t_brake = 1.0;
    sim.run(t_brake, 0.001);
    const Real V_after = vx(sim);
    const Real decel = (V_before - V_after) / t_brake;

    // The braking curve at the static load, swept every 1e-4 in slip.
    const MagicFormulaTyre tyre(vp.tyre);
    const Real Fz_static = vp.weight_per_wheel();
    MagicFormulaInput in;
    in.Fz = Fz_static;
    in.Vcx = 20.0;
    Real peak = 0.0;
    for (int i = 0; i <= 10000; ++i) {
        in.kappa = -1e-4 * i;
        peak = std::max(peak, -tyre.evaluate(in).Fx);
    }
    in.kappa = -1.0;
    const Real mu_peak  = peak / Fz_static;
    const Real mu_slide = -tyre.evaluate(in).Fx / Fz_static;

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

TEST_CASE("Drivetrain: braking to a stop locks the wheels, and the car stays stopped",
          "[drivetrain][braking]")
{
    // From 5 m/s with full pedal. The front brakes (1950 Nm each) beat the
    // front tyres' sliding torque (about 0.65 x 5.6 kN x 0.34 m = 1.2 kNm), so
    // the front wheels stop and are locked at exactly zero (task 4.1's
    // dry-friction lock); the rear wheels roll on until the car stops, about
    // v / (0.9 mu_slide g) = 0.9 s later, and are locked then. After that
    // nothing drives the car: what remains is the body's motion on its
    // suspension, which decays at the suspension's damping rate, so the
    // residual speed must at least halve from one half second to the next and
    // end below 0.1 mm/s, stopped by any practical measure.
    //
    // Since task 7.4 the tyres lag: the sliding front tyres stop with their
    // tread's deflection at the limit of (8.112), three times the slip of
    // full sliding over the relaxation length, about 3.5 cm, and give it
    // back as the car stops. The car rocks back past the rear tyres' balance,
    // and the reversed force on a rear tyre briefly exceeds its brake's 1050 N
    // m: the wheel slips back, up to 11.7 rad/s for a few milliseconds
    // (measured, every method), and is locked again. The car comes to rest
    // as before: at 3 s within 4.1e-5 m/s by every method (measured). So the
    // guard against wheels turning backwards (the brake's direction, task
    // 4.2) holds while the car still moves forwards.
    //
    // The same with Dormand-Prince (task 4.2), whose error control would
    // shrink its steps without end if the brake's sign flipped within a step
    // as a spin crossed zero: the brake keeps its direction through each step.
    // And with generalised-alpha (task 4.3), whose history starts afresh when
    // a lock changes a spin between steps, and with SDIRK4 and Radau IIA (task
    // 4.4), whose error control crosses the stops as Dormand-Prince's does.
    // And with linearly implicit Euler (task 4.5), whose drivetrain spins are
    // coupled to the hubs through the tyres in one linear solve.
    using namespace mbd;

    for (const auto method : {kernel::Integrator::RK4, kernel::Integrator::DormandPrince,
                              kernel::Integrator::GeneralizedAlpha, kernel::Integrator::SDIRK4,
                              kernel::Integrator::RadauIIA, kernel::Integrator::LinearlyImplicitEuler}) {
        INFO((method == kernel::Integrator::RK4                ? "RK4"
              : method == kernel::Integrator::DormandPrince    ? "Dormand-Prince"
              : method == kernel::Integrator::GeneralizedAlpha ? "generalised-alpha"
              : method == kernel::Integrator::SDIRK4           ? "SDIRK4"
              : method == kernel::Integrator::RadauIIA         ? "Radau IIA"
                                                               : "linearly implicit Euler"));
        kernel::System sys;
        VehicleParams vp;
        auto vm = build_simple_vehicle(sys, vp);
        kernel::Simulator sim(sys);
        sim.method = method;
        set_vehicle_equilibrium(sim, vm);
        sim.v(3) = 5.0;
        sim.initialize();

        Drivetrain dt;
        dt.initialize(sim, vm);
        dt.connect(sim, vm);
        sim.run(0.3, 0.001);

        dt.brake = 1.0;
        Real slowest_spin = 0.0;   // while the car still moves forwards
        Real slowest_any = 0.0;    // at any time
        int slowest_step = 0, slowest_corner = -1;
        Real fastest_late = 0.0, fastest_last = 0.0;   // |vx| over 2.0-2.5 s and 2.5-3.0 s
        for (int i = 1; i <= 3000; ++i) {
            sim.step(0.001);
            for (int c = 0; c < 4; ++c) slowest_any = std::min(slowest_any, dt.wheel_omega[c]);
            for (int c = 0; c < 4 && vx(sim) > 0.5; ++c) {
                if (dt.wheel_omega[c] < slowest_spin) {
                    slowest_spin = dt.wheel_omega[c];
                    slowest_step = i;
                    slowest_corner = c;
                }
            }
            if (i > 2000 && i <= 2500) fastest_late = std::max(fastest_late, std::abs(vx(sim)));
            if (i > 2500) fastest_last = std::max(fastest_last, std::abs(vx(sim)));
        }
        INFO("largest speed over 2.0-2.5 s " << fastest_late << " m/s, over 2.5-3.0 s " << fastest_last
             << " m/s, at the end " << vx(sim) << " m/s; " << sim.integrator_stats().steps << " steps, "
             << sim.integrator_stats().rejected << " rejected; slowest spin " << slowest_spin << " rad/s, corner "
             << slowest_corner << ", " << slowest_step << " ms into the braking; at any time " << slowest_any
             << " rad/s");
        CHECK(fastest_last < 0.5 * fastest_late);
        CHECK(std::abs(vx(sim)) < 1e-4);
        CHECK(slowest_spin >= 0.0);   // no wheel turned backwards while the car moved forwards
        for (int c = 0; c < 4; ++c) {
            INFO("corner " << c);
            CHECK(dt.wheel_locked[c]);
            CHECK(dt.wheel_omega[c] == 0.0);
        }
    }
}

TEST_CASE("Drivetrain: a car held on its brakes on a slope stays put",
          "[drivetrain][braking][slope]")
{
    // The plan's "done when" for task 7.4, its last part. The simple car on a
    // 10 % slope, facing up it, its brakes full on from rest: a level road
    // under gravity turned by beta = atan(0.1), the same problem turned.
    // Locked, each tyre holds by its tread's deflection (task 7.4, D64): at
    // rest a spring of the carcass's stiffness cx = Kxk / sigma_kappa (the
    // Magic Formula's shifts are faded out at rest). The car rolls back until
    // the four carry the weight's component, F = m g sin(beta) = 1523 N, and
    // pitches nose up on its springs under the moment of F about the centre
    // of gravity, at the height h = 0.508 m above the road, less that of the
    // wheels' own weight along the slope at their centres' height z_w:
    //     M = F h - 4 m_w g sin(beta) (h - z_w),   theta = M / k_pitch,
    // k_pitch = 2 k (a^2 + b^2), k the spring and the tyre in series. A
    // locked car pitches as a body about its tyres' contacts: the tread
    // moves by the centre's displacement less theta h. So the centre moves
    // back by
    //     x_static = F / (4 cx) + h M / k_pitch,
    // 1.07 + 2.31 = 3.38 mm, cx at the static load (the axles' loads differ
    // on the slope by 4 %, and cx follows them by pcfx1 dfz: 0.7 %, which
    // cancels across the axles). While the car rings down the deflections
    // also relax over the distance travelled to and fro (-|Vx| u / sigma),
    // which can only let it further back, by u / sigma (0.4 %) of that
    // distance: allowed from 2 % short (the loads' cos(beta), R_eff against
    // the loaded radius) to 10 % beyond. The low-speed damping, k_V_low0 =
    // 770 N s/m a tyre, is 3 % of critical against the car's mass on the four
    // springs, but the suspension's damping takes the pitch: the means over
    // 5-6 s and 9-10 s differ by less than 1e-5 m, and the speed at 10 s is
    // under 1e-5 m/s.
    //
    // Without the lag (the slips over max(|Vx|, V_low), as the tyre was
    // before), a locked tyre's force is Kxk (-Vx) / V_low: a damper. The car
    // creeps down at the speed where the four carry the same load,
    //     v_creep = m g sin(beta) V_low / (4 Kxk),
    // about 4.6 mm/s (F7's finding, with the tyre of then). Measured over
    // 5-10 s; allowed 10 % (the axles' loads, and the shifts, which fade
    // in with the speed's square: 5e-5 of them at this speed).
    using namespace mbd;
    const Real beta = std::atan(0.1);
    struct Run {
        Real mean_early, mean_late, speed_end, moved_back;
        bool locked;
    };
    auto run = [&](bool lag) {
        kernel::System sys;
        VehicleParams vp;
        auto vm = build_simple_vehicle(sys, vp);
        sys.model.gravity = g_accel * Vec3(-std::sin(beta), 0.0, -std::cos(beta));   // after: the builder sets it
        for (FullTireForce* t : vm.tires) t->lag = lag;
        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vm);
        sim.initialize();
        const Real x0 = sim.states()[static_cast<std::size_t>(vm.chassis_body)].p_WB.x();

        Drivetrain dt;
        dt.initialize(sim, vm);
        dt.throttle = 0.0;
        dt.brake = 1.0;
        dt.connect(sim, vm);

        Run r{0.0, 0.0, 0.0, 0.0, true};
        for (int i = 1; i <= 10000; ++i) {
            sim.step(0.001);
            const Real x = sim.states()[static_cast<std::size_t>(vm.chassis_body)].p_WB.x();
            if (i > 5000 && i <= 6000) r.mean_early += x / 1000.0;
            if (i > 9000) r.mean_late += x / 1000.0;
        }
        r.speed_end = vx(sim);
        r.moved_back = x0 - r.mean_late;
        for (int c = 0; c < 4; ++c) r.locked = r.locked && dt.wheel_locked[c];
        return r;
    };

    VehicleParams vp;
    const MagicFormulaTyre tyre(vp.tyre);
    const MagicFormulaOutput at_rest = tyre.characteristics(vp.weight_per_wheel(), 0.0);
    REQUIRE(at_rest.Fx == 0.0);   // faded out at rest
    const Real load = vp.total_mass() * g_accel * std::sin(beta);
    const Real cx = at_rest.Kxk / at_rest.sigma_kappa;
    const Real h = vp.chassis_height_eq(), z_w = vp.wheel_center_height_eq();
    const Real k = vp.k_susp * vp.tyre.VERTICAL_STIFFNESS / (vp.k_susp + vp.tyre.VERTICAL_STIFFNESS);
    const Real k_pitch = 2.0 * k * (vp.front_axle_x * vp.front_axle_x + vp.rear_axle_x * vp.rear_axle_x);
    const Real M = load * h - 4.0 * vp.wheel_mass * g_accel * std::sin(beta) * (h - z_w);
    const Real x_static = load / (4.0 * cx) + h * M / k_pitch;
    const Real v_creep = load * vp.tyre.VXLOW / (4.0 * at_rest.Kxk);

    const Run held = run(true);
    INFO("with the lag: rolled back " << held.moved_back << " m (by hand " << x_static << "), means "
         << held.mean_early << " and " << held.mean_late << " m, speed at 10 s " << held.speed_end << " m/s");
    CHECK(held.locked);
    CHECK(std::abs(held.mean_late - held.mean_early) < 1e-5);
    CHECK(std::abs(held.speed_end) < 1e-5);
    CHECK(held.moved_back > 0.98 * x_static);
    CHECK(held.moved_back < 1.10 * x_static);

    const Run crept = run(false);
    const Real creep = (crept.mean_early - crept.mean_late) / 4.0;
    INFO("without the lag: crept " << creep << " m/s (by hand " << v_creep << ")");
    CHECK(crept.locked);
    CHECK_THAT(creep, WithinRel(v_creep, 0.1));
}

TEST_CASE("Drivetrain: Dormand-Prince reproduces RK4's braking deceleration",
          "[drivetrain][braking][integrators]")
{
    // The half-pedal braking of the hand-calculation test, measured with both
    // integrators. RK4's result no longer depends on the step (identical to
    // nine digits at 1, 0.5 and 0.25 ms, task 4.1), so it is exact to about
    // 1e-8; Dormand-Prince, fifth order with its steps at most the 1 ms of
    // step() and error control at its default tolerances, must agree with it
    // to well within 1e-6. A different method of a different order reaching
    // the same number checks both.
    using namespace mbd;

    auto deceleration = [](kernel::Integrator method, kernel::IntegratorStats& stats) {
        kernel::System sys;
        VehicleParams vp;
        auto vm = build_simple_vehicle(sys, vp);
        kernel::Simulator sim(sys);
        sim.method = method;
        set_vehicle_equilibrium(sim, vm);
        sim.v(3) = 20.0;
        sim.initialize();
        Drivetrain dt;
        dt.params.layout = DriveLayout::RWD;
        dt.initialize(sim, vm);
        dt.connect(sim, vm);
        sim.run(0.3, 0.001);
        dt.brake = 0.5;
        sim.run(0.5, 0.001);
        const Real V1 = vx(sim);
        sim.run(1.0, 0.001);
        stats = sim.integrator_stats();
        return V1 - vx(sim);
    };
    kernel::IntegratorStats rk4_stats, dp_stats;
    const Real a_rk4 = deceleration(kernel::Integrator::RK4, rk4_stats);
    const Real a_dp = deceleration(kernel::Integrator::DormandPrince, dp_stats);
    INFO("RK4 " << std::setprecision(10) << a_rk4 << " m/s^2 (" << rk4_stats.evaluations
                << " evaluations), Dormand-Prince " << a_dp << " m/s^2 (" << dp_stats.steps << " steps, "
                << dp_stats.rejected << " rejected, " << dp_stats.evaluations << " evaluations)");
    CHECK_THAT(a_dp, WithinRel(a_rk4, 1e-6));
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

    // Only the tyres' rolling resistance takes energy out of this coasting
    // car (no drag, no engine braking; task 7.4). Each tyre's moment is the
    // Magic Formula's (50), R0 Fz0 (qsy1 + qsy3 V / V0 + qsy4 (V / V0)^4)
    // (Fz / Fz0)^qsy7, against the spin; on the wheel it is a force at the
    // arm r = R - d, which the car and the wheels' inertia share: a = 4 My /
    // r / (m + 4 I / (R_eff r)). With the Appendix 3 tyre at 15 m/s and Fz =
    // m g / 4 = 3825.9 N: My = 0.3135 x 4000 x 0.0084362 x 0.96073 = 10.16 N
    // m, r = 0.29525 m, so a = 137.7 / 1647.5 = 0.0836 m/s^2: 1.11 % of the
    // speed in 2 s. The speed falls by 1.1 % in that time, and the speed
    // term with it (0.16 % of the moment); allowed 2 %.
    const Real R = vp.tyre.UNLOADED_RADIUS, d = vp.tire_deflection_eq();
    const Real r = R - d, R_eff = R - 0.5 * d;
    const Real V = 0.5 * (V_start + V_end), V0 = vp.tyre.LONGVL;
    const Real My = R * vp.tyre.FNOMIN
                  * (vp.tyre.QSY1 + vp.tyre.QSY3 * V / V0 + vp.tyre.QSY4 * std::pow(V / V0, 4.0))
                  * std::pow(vp.weight_per_wheel() / vp.tyre.FNOMIN, vp.tyre.QSY7);
    const Real I = 0.5 * vp.wheel_mass * R * R;
    const Real a = 4.0 * My / r / (vp.total_mass() + 4.0 * I / (R_eff * r));
    const Real speed_loss_fraction = (V_start - V_end) / V_start;
    INFO("speed lost " << speed_loss_fraction << ", by hand " << a * 2.0 / V_start);
    REQUIRE_THAT(speed_loss_fraction, WithinRel(a * 2.0 / V_start, 0.02));
}

TEST_CASE("Drivetrain: generalised-alpha rolls the car at low speed in steps of 5 ms, where RK4 cannot",
          "[drivetrain][coast][integrators][implicit]")
{
    // At 0.5 m/s, below the tyres' V_low of 1 m/s, a front wheel's spin
    // relaxes to free rolling at between 2800 and 3100 per second on this car
    // (measured: RK4, whose limit on the negative real axis is 2.785 / h,
    // keeps it at 0.9 ms but not at 1 ms). At 5 ms RK4 is far outside (about
    // 15 against 2.785): the spins cannot settle, and chatter until the tyres'
    // saturation bounds them, by more than 5 % of the speed (14 % measured).
    // Generalised-alpha, stable at any step, rolls the car on as RK4 does at
    // 0.5 ms: every wheel within 1e-3 of free rolling (RK4 at 0.5 ms stays
    // within 4e-5, the tyres' own slip as the car settles), the speed the
    // same to 1e-4. Nothing takes energy
    // out of the coasting car (as in the test above), and the method's
    // damping acts on the motions too fast for the step, not on the speed,
    // which holds to 1e-3. Since task 7.3 the spins are the wheels' joints;
    // they start 3 % fast, as they did before, to give them something to
    // settle from (measured values in this comment from before 7.3). Since
    // task 7.4 the car's tyres lag, and this stiffness is gone: the test runs
    // on the tyres without lag (build_stiff_car above).
    using namespace mbd;
    struct Run {
        Real V_start, V_end, worst;   // worst: largest |spin x rolling radius - speed| / speed
        kernel::IntegratorStats stats;
    };
    auto coast = [](kernel::Integrator method, Real h) {
        kernel::System sys;
        auto vm = build_stiff_car(sys);
        kernel::Simulator sim(sys);
        sim.method = method;
        set_vehicle_equilibrium(sim, vm);
        sim.v(3) = 0.5;
        sim.initialize();
        Drivetrain dt;
        dt.initialize(sim, vm);
        spin_off_free_rolling(sim, vm);
        dt.connect(sim, vm);
        sim.run(0.3, h);
        Run r{vx(sim), 0.0, 0.0, {}};
        const int steps = static_cast<int>(std::round(2.0 / h));
        for (int k = 0; k < steps; ++k) {
            sim.step(h);
            for (int c = 0; c < 4; ++c) {
                const Real rolling = dt.wheel_omega[c] * vm.tires[c]->get_rolling_radius();
                r.worst = std::max(r.worst, std::abs(rolling - vx(sim)) / vx(sim));
            }
        }
        r.V_end = vx(sim);
        r.stats = sim.integrator_stats();
        return r;
    };
    const Run alpha = coast(kernel::Integrator::GeneralizedAlpha, 0.005);
    const Run reference = coast(kernel::Integrator::RK4, 0.0005);
    const Run rk4 = coast(kernel::Integrator::RK4, 0.005);
    INFO("generalised-alpha at 5 ms: speed " << alpha.V_start << " -> " << alpha.V_end << " m/s, rolling within "
         << alpha.worst << "; " << alpha.stats.steps << " steps, " << alpha.stats.newton_iterations
         << " Newton iterations, " << alpha.stats.jacobians << " iteration matrices, " << alpha.stats.evaluations
         << " evaluations. RK4 at 0.5 ms: " << reference.V_end << " m/s, within " << reference.worst
         << ". RK4 at 5 ms: " << rk4.V_end << " m/s, within " << rk4.worst);
    CHECK(alpha.worst < 1e-3);
    CHECK(std::abs(alpha.V_end - alpha.V_start) < 1e-3 * alpha.V_start);
    CHECK(std::abs(alpha.V_end - reference.V_end) < 1e-4 * reference.V_end);
    CHECK(rk4.worst > 0.05);
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

    // Accelerate with full throttle and watch every gear change. The gearbox
    // decides at the end of each step, on the spins of that state (the
    // drivetrain's end_step), so each shift is checked against them.
    dt.throttle = 1.0;
    Real omega_max = 0.0;
    int upshifts = 0;
    for (int i = 0; i < 10500; ++i) {
        const int gear_before = dt.current_gear;
        sim.step(0.001);
        const Real omega_after = 0.5 * (dt.wheel_omega[2] + dt.wheel_omega[3]);
        omega_max = std::max(omega_max, omega_after);

        if (dt.current_gear != gear_before) {
            // Only upward, one gear at a time, and only once the driven wheels
            // have passed the shift speed of the gear that is left.
            REQUIRE(dt.current_gear == gear_before + 1);
            REQUIRE(omega_after > shift_speed(gear_before));
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
    //     and the engine inertia seen through first gear, which task 7.3
    //     leaves out until 7.8: the car is lighter to accelerate now, and the
    //     bound holds) = 2.88 m/s^2;
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
    const Real y_before = sim.states()[vm.chassis_body].p_WB.y();

    sim.run(3.0, 0.001);
    const Real y_after = sim.states()[vm.chassis_body].p_WB.y();

    // Vehicle should turn left (positive Y)
    REQUIRE(y_after - y_before > 0.05);

    // Vehicle should still be on the ground
    REQUIRE_THAT(sim.states()[vm.chassis_body].p_WB.z(),
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

    // Free rolling at the static deflection: each tyre carries a quarter of
    // the car's weight, deflecting by W / k_z, and rolls at R_eff = R_free -
    // deflection / 2; 20 m/s over it. The spins are the wheels' joints' (the
    // chassis does not turn), to rounding.
    const Real R_eff = vp.tyre.UNLOADED_RADIUS - 0.5 * vp.tire_deflection_eq();
    const Real expected_omega = 20.0 / R_eff;

    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(dt.wheel_omega[c], WithinAbs(expected_omega, 1e-9));
        REQUIRE_THAT(sim.v(vm.spin_v[c]), WithinAbs(expected_omega, 1e-9));
    }
    REQUIRE(dt.current_gear > 1);   // 20 m/s is past first gear's middle
}

TEST_CASE("Drivetrain: linearly implicit Euler rolls the car at low speed in steps of 5 ms, one evaluation a step",
          "[drivetrain][coast][integrators][implicit][linear]")
{
    // As for generalised-alpha above: at 0.5 m/s the front spins relax at
    // about 3000 per second, 15 times past RK4's limit at 5 ms. The linearly
    // implicit step (task 4.5) is backward Euler in the spins, coupled to the
    // hubs' velocities through the tyres' slip, so the spins settle within
    // the step: every wheel within 1e-3 of free rolling, as generalised-alpha
    // keeps them. Backward Euler damps; nothing feeds the coasting car, so
    // its speed may only fall, by the method's damping of the small motions
    // left from the release (about 2e-5 of the speed measured): within 1e-3.
    // Its first order shows in the release transient it damps differently
    // from RK4 at 0.5 ms: the speed after it within 1e-3 of RK4's. Every step
    // costs one evaluation and one set of derivatives. Since task 7.3 the
    // spins are the wheels' joints; they start 3 % fast, as they did before,
    // to give them something to settle from (measured values in this comment
    // from before 7.3). Since task 7.4 it runs on the tyres without lag, as
    // the generalised-alpha test above (build_stiff_car).
    using namespace mbd;
    kernel::System sys;
    auto vm = build_stiff_car(sys);
    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::LinearlyImplicitEuler;
    set_vehicle_equilibrium(sim, vm);
    sim.v(3) = 0.5;
    sim.initialize();
    Drivetrain dt;
    dt.initialize(sim, vm);
    spin_off_free_rolling(sim, vm);
    dt.connect(sim, vm);
    const Real h = 0.005;
    sim.run(0.3, h);
    const Real V_start = vx(sim);
    Real worst = 0.0;
    for (int k = 0; k < 400; ++k) {
        sim.step(h);
        for (int c = 0; c < 4; ++c) {
            const Real rolling = dt.wheel_omega[c] * vm.tires[c]->get_rolling_radius();
            worst = std::max(worst, std::abs(rolling - vx(sim)) / vx(sim));
        }
    }
    const Real V_end = vx(sim);

    kernel::System ref_sys;
    auto ref_vm = build_stiff_car(ref_sys);
    kernel::Simulator ref(ref_sys);
    set_vehicle_equilibrium(ref, ref_vm);
    ref.v(3) = 0.5;
    ref.initialize();
    Drivetrain ref_dt;
    ref_dt.initialize(ref, ref_vm);
    spin_off_free_rolling(ref, ref_vm);
    ref_dt.connect(ref, ref_vm);
    ref.run(2.3, 0.0005);

    const auto& s = sim.integrator_stats();
    INFO("speed " << V_start << " -> " << V_end << " m/s (RK4 at 0.5 ms: " << vx(ref) << "), rolling within " << worst
                  << "; " << s.steps << " steps, " << s.evaluations << " evaluations, " << s.jacobians
                  << " sets of derivatives");
    CHECK(worst < 1e-3);
    CHECK(V_end <= V_start);
    CHECK(V_start - V_end < 1e-3 * V_start);
    CHECK(std::abs(V_end - vx(ref)) < 1e-3 * vx(ref));
    CHECK(s.evaluations == s.steps);
    CHECK(s.jacobians == s.steps);
}

TEST_CASE("Drivetrain: on the sedan every linearly implicit step does the same work, through a whole drive",
          "[drivetrain][integrators][implicit][linear]")
{
    // The plan's "done when" for task 4.5: the cost of a step does not vary.
    // The double-wishbone sedan (31 velocities, 4 of them the wheel spins
    // since task 7.3, its corners closed by constraints) is driven at full
    // stretch of its states: half
    // throttle from rest for 1 s, braking to a stop for 2 s (the wheels lock,
    // task 4.1), standing for 1 s. Every step must take exactly one
    // evaluation and one set of derivatives, whatever the state; RK4 takes
    // four evaluations, generalised-alpha from two to dozens (a new
    // iteration matrix costs 2 nv + nx evaluations). The drive must match
    // RK4's at the same step: the top speed within 1 % (a first-order method
    // at 1 ms on motions of a tenth of a second), and the car stopped at the
    // end: since task 7.4 it rocks on its tyres' treads after the stop (the
    // braking test above), lightly damped at rest, so after a second of
    // standing it still moves at 2.4 mm/s by both methods (measured): below
    // 5 mm/s, against the 1.7 m/s of the drive.
    // Time per step is measured by the benchmark (data file), not here.
    using namespace mbd;
    auto drive = [](kernel::Integrator method, long long& fewest, long long& most) {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
        tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);
        kernel::Simulator sim(sys);
        sim.method = method;
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        Drivetrain dt(tmpl.drivetrain);
        dt.initialize(sim, vh);
        dt.connect(sim, vh);
        fewest = 1 << 30;
        most = 0;
        Real top = 0.0;
        for (int k = 0; k < 4000; ++k) {
            dt.throttle = k < 1000 ? 0.5 : 0.0;
            dt.brake = k >= 1000 && k < 3000 ? 0.6 : 0.0;
            const auto before = sim.integrator_stats();
            sim.step(0.001);
            const auto& after = sim.integrator_stats();
            const long long work = (after.evaluations - before.evaluations) * 1000 + (after.jacobians - before.jacobians);
            fewest = std::min(fewest, work);
            most = std::max(most, work);
            top = std::max(top, vx(sim));
        }
        return std::make_pair(top, vx(sim));
    };
    long long lin_fewest = 0, lin_most = 0, rk4_fewest = 0, rk4_most = 0;
    const auto [lin_top, lin_end] = drive(kernel::Integrator::LinearlyImplicitEuler, lin_fewest, lin_most);
    const auto [rk4_top, rk4_end] = drive(kernel::Integrator::RK4, rk4_fewest, rk4_most);
    INFO("linearly implicit Euler: top speed " << lin_top << " m/s, at the end " << lin_end
         << "; work per step (evaluations x 1000 + derivative sets) " << lin_fewest << " to " << lin_most
         << ". RK4: top " << rk4_top << ", end " << rk4_end << "; " << rk4_fewest << " to " << rk4_most);
    CHECK(lin_fewest == 1001);
    CHECK(lin_most == 1001);
    CHECK(std::abs(lin_top - rk4_top) < 0.01 * rk4_top);
    CHECK(std::abs(lin_end) < 5e-3);
    CHECK(std::abs(rk4_end) < 5e-3);
}
