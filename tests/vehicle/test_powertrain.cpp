// Task 7.8 (D69): the powertrain. An engine with its own speed, a torque
// converter or a clutch (or a motor coupled directly), a gearbox with shift
// time and hysteresis, differentials, compliant half-shafts with an optional
// hysteretic isolator, and the reactions on the right bodies. The plan's
// test: anti-squat equals its geometric value. Expected values are closed
// forms written out here.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/forces/user_force.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/vehicle/powertrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace {

/// Four wheels of `mass` (radius 0.3 m) on spin joints on a fixed chassis,
/// without gravity: a driveline's test bench.
struct Bench {
    kernel::System sys;
    BodyIndex chassis{0};
    std::array<Powertrain::Corner, 4> corners{};
};

void build_bench(Bench& b, const std::array<Real, 4>& mass)
{
    b.sys.model.gravity = Vec3::Zero();
    b.chassis = b.sys.model.add_body(0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
                                     Transform3::Identity(),
                                     RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.5, 0.2)), "chassis");
    const std::array<Vec3, 4> at{Vec3(1.3, 0.8, 0.0), Vec3(1.3, -0.8, 0.0), Vec3(-1.3, 0.8, 0.0),
                                 Vec3(-1.3, -0.8, 0.0)};
    for (std::size_t c = 0; c < 4; ++c) {
        const BodyIndex w = add_wheel(b.sys.model, b.chassis, at[c], mass[c], 0.3, "wheel" + std::to_string(c));
        b.corners[c] = {w, b.chassis, b.chassis};
    }
}

/// Hold a wheel's spin at rest.
void hold_spin(kernel::System& sys, BodyIndex wheel)
{
    sys.constraints.push_back(std::make_shared<kernel::JointDriver>(sys.model, wheel, kernel::TimeFunction::constant(0.0)));
}

Real spin_rate(const kernel::Simulator& sim, BodyIndex wheel)
{
    return sim.v(sim.system.model.idx_v[static_cast<std::size_t>(wheel)]);
}

Real state(const kernel::Simulator& sim, const Powertrain& p, int index)
{
    std::size_t me = 0;
    while (sim.state_elements()[me] != &p) ++me;
    return sim.x(sim.state_offset(me) + index);
}

/// A motor of constant torque T (no power limit), coupled directly.
PowertrainParams direct_motor(Real T)
{
    PowertrainParams p;
    p.coupling = Coupling::Direct;
    p.engine = EngineMap::motor(T, 1e6, 1e6, 0.02);
    p.gear_ratios = {1.0};
    p.final_drive = 8.0;
    p.automatic = false;
    return p;
}

} // namespace

// ============================================================================
// Anti-squat
// ============================================================================

TEST_CASE("Powertrain: anti-squat equals its geometric value when the drive reacts on the chassis",
          "[vehicle][powertrain][antisquat]")
{
    // The plan's test of task 7.8. The sedan's rear trailing arms (pivot
    // 0.45 m ahead of the wheel centre and 0.02 m above), on a fixed
    // chassis, each wheel centre held at its height by a bump driver, no
    // gravity. The powertrain (rear-wheel drive) applies its half-shafts'
    // torque T on each rear wheel, reacting on the chassis (an inboard
    // differential); a force F_x = T / R at the contact, R = 0.3 m below the
    // wheel centre, balances the wheel's spin, as the tyre's would. The
    // drivers' multipliers then give the share of the drive force the
    // suspension carries against squat: lambda = -(the applied forces'
    // power per unit of travel) = -F_x dx_wc/dz (the torque's arm R cancels
    // the contact's), and for a vertical load at the contact -dz_c/dz. The
    // anti-squat is -(dx_wc / dz_wc) L / h (L = 2.7 m, h = 0.3 m): with the
    // wheel centre moving back by 0.02 / 0.45 per unit of rise, 0.0444 * 9 =
    // 40 %. The multipliers give it to rounding. Were the reaction on the
    // arm (as a brake's is), it would be -(dx_c / dz_c) L / h, the contact's
    // path, which moves back by (0.02 + R) / 0.45: 640 %; also to rounding.
    VehicleTemplate t = VehicleTemplate::DefaultSedan();
    t.rear_axle.suspension_type = SuspensionType::TrailingArm;
    const Real L = 2.7, h = 0.3, R = 0.3, Fx = 1000.0;
    for (const bool inboard : {true, false}) {
        kernel::System sys;
        const CornerRig left = build_corner_rig(sys, t, 2);
        sys.model.gravity = Vec3::Zero();   // after the rig, which sets it
        // The right rear corner on the same chassis, and its own driver.
        const Vec3 rw(-t.rear_axle_x, -t.rear_axle.half_track, 0.0);
        const CornerHandle right =
            detail::build_linkage_corner(sys, left.chassis_body, rw, t.rear_axle, true, left.chassis_body, "R");
        const std::size_t right_driver = sys.constraints.size();
        sys.constraints.push_back(point_height_driver(right.upright_body, Vec3::Zero(), 2, 0.0));
        const std::array<CornerHandle, 2> rear{left.corners[0], right};

        PowertrainParams p = direct_motor(0.0);
        Powertrain pt(p);
        std::array<Powertrain::Corner, 4> corners{};
        corners[0] = corners[1] = {rear[0].wheel_body, rear[0].upright_body, left.chassis_body};   // undriven
        for (std::size_t s = 0; s < 2; ++s) {
            corners[2 + s] = {rear[s].wheel_body, rear[s].upright_body,
                              inboard ? left.chassis_body : rear[s].upright_body};
        }
        // The contact force on each rear wheel, R below its centre, forward.
        for (const CornerHandle& c : rear) {
            const BodyIndex w = c.wheel_body;
            sys.force_elements.push_back(std::make_shared<UserForce>(
                "contact", std::vector<BodyIndex>{w},
                [w, R, Fx](const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) {
                    const RigidBodyState& s = states[static_cast<std::size_t>(w)];
                    const Vec3 F = Fx * Vec3::UnitX();
                    forces[static_cast<std::size_t>(w)].f_W += F;
                    forces[static_cast<std::size_t>(w)].tau_W += Vec3(0.0, 0.0, -R).cross(F);
                    (void)s;
                }));
        }
        kernel::Simulator sim(sys);
        sim.initialize();
        // At rest, before the powertrain takes its driveline's speeds from
        // the wheels: the projection gave the arms the drivers' rate and the
        // wheels' spins with it, which the shafts' damping would turn into
        // torque.
        sim.v.setZero();
        sim.refresh();
        pt.connect(sim, left.chassis_body, corners);
        // Twist each rear half-shaft to carry T = F_x R.
        std::size_t me = 0;
        while (sim.state_elements()[me] != &pt) ++me;
        for (int c = 2; c < 4; ++c) sim.x(sim.state_offset(me) + pt.twist_index(c)) = Fx * R / p.shaft_stiffness;
        const kernel::Loads loads = kernel::compute_loads(sim);
        auto multiplier = [&](std::size_t constraint) {
            Index row = 0;
            for (std::size_t i = 0; i < constraint; ++i) row += sys.constraints[i]->size();
            return loads.lambda(row);
        };
        const Real lambda_x = multiplier(left.drivers[0]);
        CHECK_THAT(multiplier(right_driver), WithinRel(lambda_x, 1e-12));   // the mirror image
        // The geometry: the wheel centre's and the contact's paths.
        Kinematics k(sys);
        REQUIRE(k.solve(50, 1e-12));
        const VecX v = k.velocities();
        kernel::Data data(sys.model);
        kernel::forward_kinematics(sys.model, data, k.q, v);
        const RigidBodyState arm = kernel::body_state(data, rear[0].upright_body);
        const Real dx_wc = arm.v_WB.x(), dz_wc = arm.v_WB.z();
        const Real dx_c = (arm.v_WB + arm.w_WB.cross(Vec3(0.0, 0.0, -R))).x();
        const Real geometric = (inboard ? -dx_wc : -dx_c) / dz_wc * L / h;
        // lambda_z for a unit vertical load at the contact: -dz_c, here -dz_wc
        // (the arm turns about Y, which moves the contact only along X).
        const Real lambda_z = -dz_wc;
        const Real measured = -(lambda_x / Fx) / lambda_z * L / h;
        INFO((inboard ? "inboard" : "outboard") << ": anti-squat " << 100.0 * measured << " % from the multipliers, "
             << 100.0 * geometric << " % from the geometry");
        CHECK_THAT(measured, WithinRel(geometric, 1e-10));
        CHECK_THAT(geometric, WithinRel(inboard ? 0.02 / 0.45 * L / h : (0.02 + R) / 0.45 * L / h, 1e-10));
    }
}

// ============================================================================
// The driveline on the bench
// ============================================================================

TEST_CASE("Powertrain: a torque converter at stall turns the engine at K sqrt(T) and multiplies its torque",
          "[vehicle][powertrain][converter]")
{
    // A constant full-load torque T = 300 N m (no drag, no idle control),
    // the rear wheels held: the turbine at rest, the converter's pump takes
    // (w_e / K)^2, so the engine settles at w_e = K sqrt(T) = 12 sqrt(300) =
    // 207.846 rad/s, and the turbine gives TR_0 T = 600 N m; each half-shaft
    // carries half of eta g 600 = 0.92 * 12.25 * 600 / 2 = 3381 N m (first
    // gear, 3.5 times the final drive's 3.5). The engine's time constant is
    // I_e / (2 w_e / K^2) = 0.069 s, the shafts' modes damped within 0.1 s:
    // after 2 s the transients are below 1e-12; allow 1e-9 relative. With a
    // Bouc-Wen isolator in each shaft (k_h = 4000 N m/rad, z_u = 1 / (beta +
    // gamma) = 0.02 rad, so at most 80 N m), the shaft's torque is k a + k_h
    // z at rest, z within its bound and near it (the twist rose far past
    // 1 / (beta + gamma), then the oscillations' unloading branches eased it
    // a little: z is the path's, not z_u).
    for (const bool isolator : {false, true}) {
        Bench b;
        build_bench(b, {{20.0, 20.0, 20.0, 20.0}});
        hold_spin(b.sys, b.corners[2].wheel);
        hold_spin(b.sys, b.corners[3].wheel);
        PowertrainParams p;
        p.engine.full_load = Curve::linear(0.0, 300.0);
        p.engine.drag = Curve();
        p.engine.idle_speed = 0.0;
        p.engine.inertia = 0.2;
        p.automatic = false;
        if (isolator) p.isolator = BoucWen{4000.0, 1.0, 30.0, 20.0, 1.0};
        Powertrain pt(p);
        pt.throttle = 1.0;
        kernel::Simulator sim(b.sys);
        sim.initialize();
        pt.connect(sim, b.chassis, b.corners);
        sim.run(2.0, 2e-4);
        const Real w_e = state(sim, pt, pt.engine_index());
        const Real shaft = 0.5 * 0.92 * 12.25 * 2.0 * 300.0;
        INFO("engine " << w_e << " rad/s; shafts " << pt.shaft_torque[2] << ", " << pt.shaft_torque[3] << " N m");
        CHECK_THAT(w_e, WithinRel(12.0 * std::sqrt(300.0), 1e-9));
        CHECK_THAT(pt.shaft_torque[2], WithinRel(shaft, 1e-9));
        CHECK_THAT(pt.shaft_torque[3], WithinRel(shaft, 1e-9));
        const Real twist = state(sim, pt, pt.twist_index(2));
        const Real z = isolator ? state(sim, pt, pt.twist_index(2) + 2) : 0.0;
        INFO("twist " << twist << " rad, Bouc-Wen z " << z << " rad");
        CHECK_THAT(pt.shaft_torque[2], WithinRel(p.shaft_stiffness * twist + 4000.0 * z, 1e-9));
        if (isolator) {
            CHECK(z <= 0.02);
            CHECK(z > 0.9 * 0.02);
        }
    }
}

TEST_CASE("Powertrain: an open differential shares torque, a locked one speed, as their rigid limits say",
          "[vehicle][powertrain][differential]")
{
    // A motor of 100 N m coupled directly (8:1) to the rear axle, its rear
    // wheels free with inertias J1 = 0.9 and J2 = 1.8 kg m^2 (20 and 40 kg,
    // radius 0.3 m). Once the shafts' oscillations have died (their slowest
    // mode, 0.9 kg m^2 on 1e4 N m/rad with 20 N m s/rad, decays at 11 s^-1:
    // e^-16.5 = 7e-8 of its start after 1.5 s, its start of the order of the
    // accelerations), the shafts' twists are steady and the wheels
    // accelerate as on rigid shafts. With A = J_c + 2 J_g + (J_in + I_m) g^2 and B =
    // 2 J_g + J_s, the carrier's and spider's speeds w_c, w_s and the wheels
    // w_c +- w_s:
    //   open:   [[A + J1 + J2, J1 - J2], [J1 - J2, B + J1 + J2]] [w_c', w_s']
    //           = [eta g T, 0];
    //   locked: w_c' = eta g T / (A + J1 + J2), both wheels;
    //   limited slip, viscous c_v = 10 N m s: the coupling's -4 c_v w_s on
    //           the spider's row; at its steady state (its time constant
    //           (m22 - m12^2 / m11) / (4 c_v) = 0.065 s) w_s' = 0, both wheels
    //           at eta g T / m11 and w_L - w_R = 2 w_s = -m12 eta g T /
    //           (2 c_v m11).
    // The wheels' accelerations, from their spins between 1.5 and 2 s:
    // allow 1e-6 relative.
    for (const Differential diff : {Differential::Open, Differential::Locked, Differential::LimitedSlip}) {
        Bench b;
        build_bench(b, {{20.0, 20.0, 20.0, 40.0}});
        PowertrainParams p = direct_motor(100.0);
        p.differential = diff;
        p.lsd_viscous = 10.0;
        Powertrain pt(p);
        pt.throttle = 1.0;
        kernel::Simulator sim(b.sys);
        sim.initialize();
        pt.connect(sim, b.chassis, b.corners);
        sim.run(1.5, 2e-4);
        const std::array<Real, 2> w0{spin_rate(sim, b.corners[2].wheel), spin_rate(sim, b.corners[3].wheel)};
        sim.run(0.5, 2e-4);
        const std::array<Real, 2> a{(spin_rate(sim, b.corners[2].wheel) - w0[0]) / 0.5,
                                    (spin_rate(sim, b.corners[3].wheel) - w0[1]) / 0.5};
        const Real J1 = 0.5 * 20.0 * 0.09, J2 = 0.5 * 40.0 * 0.09, g = 8.0;
        const Real A = p.carrier_inertia + 2.0 * p.side_gear_inertia + (p.input_inertia + p.engine.inertia) * g * g;
        const Real B = 2.0 * p.side_gear_inertia + p.spider_inertia;
        const Real T = p.efficiency * g * 100.0;
        std::array<Real, 2> expected{};
        if (diff == Differential::Open) {
            const Real m11 = A + J1 + J2, m12 = J1 - J2, m22 = B + J1 + J2, det = m11 * m22 - m12 * m12;
            const Real wc = m22 * T / det, ws = -m12 * T / det;
            expected = {{wc + ws, wc - ws}};
        } else if (diff == Differential::Locked) {
            expected = {{T / (A + J1 + J2), T / (A + J1 + J2)}};
        } else {
            const Real m11 = A + J1 + J2, m12 = J1 - J2;
            expected = {{T / m11, T / m11}};
            const Real gap = spin_rate(sim, b.corners[2].wheel) - spin_rate(sim, b.corners[3].wheel);
            INFO("limited slip: the wheels' speeds differ by " << gap << " rad/s, the steady state "
                 << -m12 * T / (2.0 * 10.0 * m11));
            CHECK_THAT(gap, WithinRel(-m12 * T / (2.0 * 10.0 * m11), 1e-6));
        }
        INFO((diff == Differential::Open ? "open" : diff == Differential::Locked ? "locked" : "limited slip")
             << ": wheels accelerate at " << a[0] << " and " << a[1]
             << " rad/s^2, the rigid limit " << expected[0] << " and " << expected[1]);
        CHECK_THAT(a[0], WithinRel(expected[0], 1e-6));
        CHECK_THAT(a[1], WithinRel(expected[1], 1e-6));
    }
}

TEST_CASE("Powertrain: engine braking follows the drag curve, and a motor's torque its power limit",
          "[vehicle][powertrain][engine]")
{
    // The template's engine (400 N m), declutched, throttle shut, from 5000
    // rpm (523.6 rad/s): its drag is linear, a + b w with a = -20 N m and b =
    // -0.07 * 400 / 733.04 = -0.038197 N m s, so I w' = a + b w gives w(t) =
    // (w0 + a / b) e^(b t / I) - a / b (I = 0.15 kg m^2): after 0.5 s,
    // 437.53 rad/s, above idle. RK4 at 0.2 ms: to 1e-9 relative.
    {
        Bench b;
        build_bench(b, {{20.0, 20.0, 20.0, 20.0}});
        PowertrainParams p;
        p.coupling = Coupling::Clutch;
        p.engine = EngineMap::from(EngineParams{});
        p.automatic = false;
        Powertrain pt(p);
        pt.clutch = 0.0;
        kernel::Simulator sim(b.sys);
        sim.initialize();
        pt.connect(sim, b.chassis, b.corners);
        std::size_t me = 0;
        while (sim.state_elements()[me] != &pt) ++me;
        const Real w0 = 5000.0 * 2.0 * pi / 60.0;
        sim.x(sim.state_offset(me) + pt.engine_index()) = w0;
        sim.run(0.5, 2e-4);
        const Real a = -20.0, bb = -0.07 * 400.0 / (7000.0 * 2.0 * pi / 60.0), I = 0.15;
        const Real expected = (w0 + a / bb) * std::exp(bb * 0.5 / I) - a / bb;
        INFO("engine after 0.5 s: " << state(sim, pt, pt.engine_index()) << " rad/s, closed form " << expected);
        CHECK_THAT(state(sim, pt, pt.engine_index()), WithinRel(expected, 1e-9));
    }
    // A motor of 300 N m to 600 rad/s, then 180 kW: at 900 rad/s, 200 N m;
    // its regeneration at zero throttle 30 % of the torque available (-90 N
    // m, -60 at 900 rad/s); half throttle there, 0.5 * 200 - 0.5 * 60.
    const EngineMap m = EngineMap::motor(300.0, 600.0, 1500.0, 0.05, 0.3);
    CHECK_THAT(m.torque(300.0, 1.0), WithinRel(300.0, 1e-15));
    CHECK_THAT(m.torque(900.0, 1.0), WithinRel(200.0, 1e-15));
    CHECK_THAT(m.torque(300.0, 0.0), WithinRel(-90.0, 1e-15));
    CHECK_THAT(m.torque(900.0, 0.0), WithinRel(-60.0, 1e-15));
    CHECK_THAT(m.torque(900.0, 0.5), WithinRel(0.5 * 200.0 - 0.5 * 60.0, 1e-15));
}

TEST_CASE("Powertrain: the gearbox shifts at its speeds, through zero torque, without hunting",
          "[vehicle][powertrain][gearbox]")
{
    // A clutch engaged, full throttle, the rear wheels heavy (as a car's
    // mass would make them: 1500 kg each, 67.5 kg m^2): the engine climbs to
    // the upshift speed, 6500 rpm. At that step's end the shift begins: the
    // gearbox's torque falls to zero at its middle, where the ratio changes
    // to second, and returns at its end (0.3 s). The engine then turns at
    // about 6500 * 2.3 / 3.5 = 4271 rpm, above the downshift speed of 2000:
    // it does not shift back (the hysteresis).
    Bench b;
    build_bench(b, {{20.0, 20.0, 1500.0, 1500.0}});
    PowertrainParams p;
    p.coupling = Coupling::Clutch;
    p.engine = EngineMap::from(EngineParams{});
    Powertrain pt(p);
    pt.throttle = 1.0;
    kernel::Simulator sim(b.sys);
    sim.initialize();
    pt.connect(sim, b.chassis, b.corners);
    Real t_shift = -1.0, t_second = -1.0, least = 1e9;
    int lowest_after = 99;
    for (int i = 0; i < 40000 && sim.time < 20.0; ++i) {
        sim.step(5e-4);
        if (t_shift < 0.0 && pt.shifting()) t_shift = sim.time;
        if (t_shift > 0.0 && t_second < 0.0 && pt.gear == 2) t_second = sim.time;
        if (pt.shifting()) least = std::min(least, std::abs(pt.gearbox_torque));
        if (t_second > 0.0) lowest_after = std::min(lowest_after, pt.gear);
        if (t_second > 0.0 && sim.time > t_second + 1.0) break;
    }
    INFO("shift began at " << t_shift << " s, second gear at " << t_second << " s; least gearbox torque in the "
         << "shift " << least << " N m");
    REQUIRE(t_shift > 0.0);
    CHECK_THAT(t_second - t_shift, WithinAbs(0.15, 5e-4 + 1e-12));
    CHECK(least < 0.01 * 0.92 * 12.25 * 400.0);   // through zero, to the step's resolution
    CHECK(lowest_after == 2);
}

// ============================================================================
// On a car
// ============================================================================

TEST_CASE("Powertrain: a car accelerates from rest through its torque converter", "[vehicle][powertrain][car]")
{
    // The sports car (double wishbones, rear-wheel drive) with the
    // powertrain instead of the template's drivetrain, from rest at full
    // throttle for 2 s (RK4 at 0.5 ms): it moves forward, the engine stays
    // between idle and redline, and its half-shafts carry equal torques
    // (straight ahead, an open differential, a symmetric car).
    const VehicleTemplate t = VehicleTemplate::SportsCar();
    kernel::System sys;
    VehicleHandle vh = build_vehicle(sys, t);
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    PowertrainParams p;
    p.engine = EngineMap::from(t.drivetrain.engine);
    Powertrain pt(p);
    pt.connect(sim, vh);
    pt.throttle = 1.0;
    sim.run(2.0, 5e-4);
    const auto& s = sim.states()[static_cast<std::size_t>(vh.chassis_body)];
    const Real speed = s.v_WB.dot(s.q_WB * Vec3::UnitX());
    const Real w_e = state(sim, pt, pt.engine_index());
    INFO("after 2 s: " << speed << " m/s, engine " << w_e << " rad/s in gear " << pt.gear << "; shafts "
         << pt.shaft_torque[2] << " and " << pt.shaft_torque[3] << " N m");
    CHECK(speed > 1.0);
    CHECK(w_e > p.engine.idle_speed);
    CHECK(w_e < p.engine.redline_speed);
    CHECK_THAT(pt.shaft_torque[2], WithinRel(pt.shaft_torque[3], 1e-6));
}

TEST_CASE("Powertrain: what it cannot model is refused", "[vehicle][powertrain]")
{
    auto error_of = [](PowertrainParams p) {
        try {
            Powertrain pt(p);
        } catch (const MbdError& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    PowertrainParams p;
    p.gear_ratios = {};
    CHECK_THAT(error_of(p), ContainsSubstring("MBD-V006"));
    p = PowertrainParams{};
    p.converter_coupling_point = 1.0;
    CHECK_THAT(error_of(p), ContainsSubstring("MBD-V006"));
    p = PowertrainParams{};
    p.front_share = 1.5;
    CHECK_THAT(error_of(p), ContainsSubstring("MBD-V006"));
}
