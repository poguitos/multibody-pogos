// Task 7.9 (D70): the brakes. Pedal to pressure (lagging) to torque at each
// wheel, the axles' bias, a proportioning valve, wheel lock, an anti-lock
// controller. The plan's test: the stopping distance matches a hand
// calculation.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include "mbd/forces/road.hpp"
#include "mbd/forces/simple_tyres.hpp"
#include "mbd/vehicle/brakes.hpp"
#include "mbd/vehicle/vehicle.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace {

Real forward_speed(const kernel::Simulator& sim, const VehicleModel& vm)
{
    const RigidBodyState& s = sim.states()[static_cast<std::size_t>(vm.chassis_body)];
    return s.v_WB.dot(s.q_WB * Vec3::UnitX());
}

/// The simple car at speed v0, its wheels rolling.
void start_at(kernel::Simulator& sim, const VehicleModel& vm, Real v0)
{
    set_vehicle_equilibrium(sim, vm);
    sim.v(sim.system.model.idx_v[static_cast<std::size_t>(vm.chassis_body)] + 3) = v0;
    sim.refresh();
    roll_wheels(sim, vm);
    sim.initialize();
}

struct Stop {
    Real distance{0.0}, time{0.0};
    int steps{0}, locked_steps{0};
};

/// Brake until the car's speed falls below 0.01 m/s.
Stop brake_to_rest(kernel::Simulator& sim, const VehicleModel& vm, const BrakeSystem& b, Real h)
{
    Stop s;
    const Real x0 = sim.states()[static_cast<std::size_t>(vm.chassis_body)].p_WB.x();
    const Real t0 = sim.time;
    while (forward_speed(sim, vm) > 0.01 && sim.time < t0 + 20.0) {
        sim.step(h);
        ++s.steps;
        if (forward_speed(sim, vm) > 2.0 && (b.locked[0] || b.locked[1] || b.locked[2] || b.locked[3])) ++s.locked_steps;
    }
    s.distance = sim.states()[static_cast<std::size_t>(vm.chassis_body)].p_WB.x() - x0;
    s.time = sim.time - t0;
    return s;
}

} // namespace

TEST_CASE("Brakes: the stopping distance matches the hand calculation", "[vehicle][brakes]")
{
    // The plan's test of task 7.9. The simple car (1400 kg body, four 40 kg
    // wheels, I = m_w R^2 / 2 each) on linear tyres without rolling
    // resistance, stiff vertically (2e7 N/m: the loaded radius moves by 0.2
    // mm under the load's transfer) and without lag; from 20 m/s, the pedal
    // at half (6 MPa) from t = 0, the pressure rising as 1 - e^(-t / tau),
    // tau = 0.05 s. Each wheel: I w' = -T + F_x R, w = V / R, so the car's
    // deceleration is d = sum(T) / R / (m + sum(I) / R^2), T the front's
    // 1.9e-4 and the rear's 0.9e-4 N m/Pa times the pressure: d(t) = d_inf
    // (1 - e^(-t / tau)), d_inf = 2 (1140 + 540) / R / (1560 + 4 * 20) =
    // 6.53 m/s^2, below the 9.81 the tyres could give: no wheel locks. Then
    //   v(t) = v0 - d_inf (t - tau (1 - e^(-t / tau))),
    //   s(t) = v0 t - d_inf (t^2 / 2 - tau t + tau^2 (1 - e^(-t / tau))),
    // the stop where v = 0 (Newton's method). What the hand calculation
    // leaves out: the tyres' slip (steady, a few per mille, which moves the
    // wheels' speed but not the car's), their loaded radius (2e-4 of R), the
    // body's pitch (its centre of mass is the chassis's origin), and the last
    // metre per second, where the tyre's force fades (V_low = 1 m/s) and the
    // wheels lock, which at the nominal deceleration takes V_low^2 / (2 d) =
    // 0.077 m, 2.5e-3 of the 31 m. Allow 3e-3, sized to that stretch (it is
    // not a bound: a smaller deceleration there would take longer).
    VehicleParams p;
    LinearTyreParams lt;
    lt.rolling_resistance = 0.0;
    lt.vertical_stiffness = 2e7;
    p.tyre_model = std::make_shared<LinearTyre>(lt);
    kernel::System sys;
    VehicleModel vm = build_simple_vehicle(sys, p);
    for (FullTireForce* t : vm.tires) t->lag = false;
    kernel::Simulator sim(sys);
    const Real v0 = 20.0;
    start_at(sim, vm, v0);
    BrakeSystemParams bp;
    BrakeSystem b(bp);
    b.connect(sim, vm);
    b.pedal = 0.5;
    const Stop stop = brake_to_rest(sim, vm, b, 5e-4);

    const Real R = lt.free_radius, m = p.total_mass(), I = 0.5 * p.wheel_mass * R * R;
    const Real pressure = 0.5 * bp.max_pressure;
    const Real d_inf = 2.0 * (bp.front_gain + bp.rear_gain) * pressure / R / (m + 4.0 * I / (R * R));
    const Real tau = bp.response_time;
    Real t = v0 / d_inf;
    for (int i = 0; i < 50; ++i) {
        const Real v = v0 - d_inf * (t - tau * (1.0 - std::exp(-t / tau)));
        t += v / (d_inf * (1.0 - std::exp(-t / tau)));
    }
    const Real s = v0 * t - d_inf * (0.5 * t * t - tau * t + tau * tau * (1.0 - std::exp(-t / tau)));
    INFO("stopped in " << stop.distance << " m and " << stop.time << " s; by hand " << s << " m and " << t << " s");
    CHECK_THAT(stop.distance, WithinRel(s, 3e-3));
    CHECK_THAT(stop.time, WithinRel(t, 3e-3));
    CHECK(stop.locked_steps == 0);
}

TEST_CASE("Brakes: on a slippery road the wheels lock and the car slides at mu g", "[vehicle][brakes]")
{
    // Full pedal (12 MPa: 2280 N m at a front wheel, where its tyre can give
    // mu Fz R = 0.3 * 3830 * 0.3135 = 360 N m) on a road of friction 0.3: the
    // wheels lock within a few hundredths of a second and stay locked; each
    // linear tyre then gives mu Fz against the slide, so the car decelerates
    // at mu g = 2.942 m/s^2 whatever the load's transfer (the loads sum to
    // the weight). Measured over a second in the middle of the slide: allow
    // 1e-3 relative (the body's pitch oscillation from the brakes' onset,
    // decayed by then).
    VehicleParams p;
    LinearTyreParams lt;
    lt.rolling_resistance = 0.0;
    p.tyre_model = std::make_shared<LinearTyre>(lt);
    kernel::System sys;
    VehicleModel vm = build_simple_vehicle(sys, p);
    vm.set_road(std::make_shared<FlatRoad>(0.0, 0.3));
    kernel::Simulator sim(sys);
    start_at(sim, vm, 20.0);
    BrakeSystem b(BrakeSystemParams{});
    b.connect(sim, vm);
    b.pedal = 1.0;
    sim.run(1.0, 5e-4);
    const Real v1 = forward_speed(sim, vm);
    bool all_locked = true;
    for (int i = 0; i < 2000; ++i) {
        sim.step(5e-4);
        for (bool l : b.locked) all_locked = all_locked && l;
    }
    const Real d = (v1 - forward_speed(sim, vm)) / 1.0;
    INFO("deceleration " << d << " m/s^2, mu g " << 0.3 * g_accel);
    CHECK(all_locked);
    CHECK_THAT(d, WithinRel(0.3 * g_accel, 1e-3));
}

TEST_CASE("Brakes: the anti-lock controller keeps the wheels turning and shortens the stop", "[vehicle][brakes]")
{
    // The Magic Formula tyre (its friction peaking near 10 % of slip and
    // falling beyond) on a road of friction 0.3, from 20 m/s at full pedal.
    // Without the controller the wheels lock and slide; with it each wheel is
    // released when its slip passes 15 % and reapplied below 8 %, so it
    // works near the peak: the car stops in a shorter distance, and above 2
    // m/s no wheel stays locked.
    std::array<Stop, 2> stops;
    for (int abs = 0; abs < 2; ++abs) {
        VehicleParams p;
        kernel::System sys;
        VehicleModel vm = build_simple_vehicle(sys, p);
        vm.set_road(std::make_shared<FlatRoad>(0.0, 0.3));
        kernel::Simulator sim(sys);
        start_at(sim, vm, 20.0);
        BrakeSystemParams bp;
        bp.abs = abs == 1;
        BrakeSystem b(bp);
        b.connect(sim, vm);
        b.pedal = 1.0;
        stops[static_cast<std::size_t>(abs)] = brake_to_rest(sim, vm, b, 5e-4);
    }
    INFO("without the controller " << stops[0].distance << " m (" << stops[0].locked_steps << " of "
         << stops[0].steps << " steps with a wheel locked above 2 m/s); with it " << stops[1].distance << " m ("
         << stops[1].locked_steps << ")");
    CHECK(stops[0].locked_steps > stops[0].steps / 2);
    CHECK(stops[1].locked_steps == 0);
    CHECK(stops[1].distance < stops[0].distance);
}

TEST_CASE("Brakes: bias, the proportioning valve, and their refusals", "[vehicle][brakes]")
{
    // 12 MPa at full pedal; the rear's pressure past a knee of 4 MPa rises by
    // 0.3 of the master's: at half pedal (6 MPa) 4 + 0.3 * 2 = 4.6 MPa. The
    // bias at equal pressures: 1.9 / (1.9 + 0.9) = 0.679 to the front.
    BrakeSystemParams bp;
    bp.knee_pressure = 4e6;
    BrakeSystem b(bp);
    b.pedal = 0.5;
    CHECK(b.target_pressure(0) == 6e6);
    CHECK_THAT(b.target_pressure(2), WithinRel(4.6e6, 1e-15));
    b.pedal = 0.2;
    CHECK(b.target_pressure(3) == 2.4e6);
    CHECK_THAT(b.torque_at(0, 1e6) / (b.torque_at(0, 1e6) + b.torque_at(2, 1e6)), WithinRel(1.9 / 2.8, 1e-15));

    auto error_of = [](BrakeSystemParams q) {
        try {
            BrakeSystem bad(q);
        } catch (const MbdError& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    BrakeSystemParams q;
    q.response_time = 0.0;
    CHECK_THAT(error_of(q), ContainsSubstring("MBD-V007"));
    q = BrakeSystemParams{};
    q.abs_apply_slip = 0.2;
    CHECK_THAT(error_of(q), ContainsSubstring("MBD-V007"));
}
