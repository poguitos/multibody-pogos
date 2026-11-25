// The regression scenarios (plan task 5.5, D49). Each runs a model of the
// test suite deterministically: fixed steps, or a solve to a tight tolerance,
// so that a result changes only when the computation does. Together they
// cover the kernel (tree and loop dynamics, gyroscopic terms, every kind of
// integrator), the force elements, the vehicle template with its tyres and
// drivetrain, statics and linearisation, suspension kinematics and the lap
// speed profile.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/analysis/lap_speed_profile.hpp"
#include "mbd/analysis/lap_vehicle.hpp"
#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/analysis/track.hpp"
#include "mbd/forces/road.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/vehicle/drivetrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

#include "kernel/bushed_pendulum.hpp"
#include "kernel/four_bar.hpp"
#include "kernel/hmmwv.hpp"
#include "regression/regression.hpp"

namespace mbd_regression {

using namespace mbd;
using namespace mbd::kernel;

namespace {

// --- The kernel --------------------------------------------------------------

/// A planar four-bar closed in 3D by a revolute closure (three redundant
/// equations), released at rest with the crank at 0.5 rad: RK4 at 1 ms, 2 s.
Record four_bar(Real scale)
{
    mbd_test::FourBar fb;
    fb.sys.model.gravity *= scale;
    Simulator sim(fb.sys);
    sim.method = Integrator::RK4;
    sim.q = mbd_test::FourBar::configuration(0.5);
    sim.v.setZero();
    sim.initialize();
    Record r{"t [s]",
             {"crank [rad]", "coupler [rad]", "rocker [rad]", "crank rate [rad/s]", "coupler rate [rad/s]",
              "rocker rate [rad/s]"},
             {}};
    for (int k = 0; k < 40; ++k) {
        sim.run(0.05, 1e-3);
        r.add(sim.time, {sim.q(0), sim.q(1), sim.q(2), sim.v(0), sim.v(1), sim.v(2)});
    }
    return r;
}

/// A heavy top held at its tip by a spherical closure, tilted 0.5 rad and
/// spun at 30 rad/s: gyroscopic terms; generalised-alpha at 1 ms, 2 s.
Record heavy_top(Real scale)
{
    constexpr Real l = 0.1;
    System sys;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel * scale);
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.02)), "top");
    sys.constraints.push_back(spherical_closure(Marker{1, Transform3::FromTranslation(Vec3(0.0, 0.0, -l))},
                                                Marker{0, Transform3()}));
    Simulator sim(sys);
    sim.method = Integrator::GeneralizedAlpha;
    const Quat tilt(Eigen::AngleAxisd(0.5, Vec3::UnitX()));
    sim.q.head<3>() = tilt * Vec3(0.0, 0.0, l);
    sim.q.tail<4>() = tilt.coeffs();
    sim.v.setZero();
    sim.v(2) = 30.0;
    sim.initialize();
    Record r{"t [s]",
             {"x [m]", "y [m]", "z [m]", "quaternion x", "quaternion y", "quaternion z", "quaternion w",
              "spin [rad/s]"},
             {}};
    for (int k = 0; k < 40; ++k) {
        sim.run(0.05, 1e-3);
        r.add(sim.time, {sim.q(0), sim.q(1), sim.q(2), sim.q(3), sim.q(4), sim.q(5), sim.q(6), sim.v(2)});
    }
    return r;
}

/// The bushed double pendulum (1e6 N/m, 1e4 N m/rad), released level:
/// bushings, Radau IIA at fixed steps of 1 ms, 1 s.
Record bushed_pendulum(Real scale)
{
    mbd_test::BushedPendulum p(1e6, 1e4);
    p.sys.model.gravity *= scale;
    Simulator sim(p.sys);
    sim.method = Integrator::RadauIIA;
    sim.step_control.adaptive = false;
    p.release(sim);
    Record r{"t [s]", {"tip x [m]", "tip z [m]", "link 2 x [m]", "link 2 z [m]"}, {}};
    for (int k = 0; k < 20; ++k) {
        sim.run(0.05, 1e-3);
        const Vec3 tip = p.tip(sim);
        const Vec3 o2 = sim.states()[static_cast<std::size_t>(p.link2)].p_WB;
        r.add(sim.time, {tip.x(), tip.z(), o2.x(), o2.z()});
    }
    return r;
}

// --- The HMMWV ---------------------------------------------------------------

/// The HMMWV of task 5.4 from rest on its tyres at 10 mph, the front tyres
/// turning to the bump at 0.2 s and the rear ones at about 0.96 s.
Record hmmwv(Real scale, bool bushed, Integrator method, Real step, Real duration, Real t_front)
{
    mbd_test::Hmmwv car(bushed);
    car.sys.model.gravity *= scale;
    Simulator sim(car.sys);
    (void)car.start(sim, t_front, false);
    sim.method = method;
    Record r{"t [s]",
             {"chassis x [m]", "chassis height [m]", "pitch [rad]", "front left wheel height [m]",
              "rear left wheel height [m]", "front left spring [m]", "rear left spring [m]"},
             {}};
    const int samples = static_cast<int>(std::lround(duration / 0.02));
    for (int k = 0; k < samples; ++k) {
        sim.run(0.02, step);
        r.add(sim.time, {car.chassis_com(sim).x(), car.chassis_com(sim).z(), car.chassis_pitch(sim),
                         car.wheel_centre(sim, 0).z(), car.wheel_centre(sim, 2).z(), car.spring_length(sim, 0),
                         car.spring_length(sim, 2)});
    }
    return r;
}

// --- The vehicle template ----------------------------------------------------

/// The chassis's placement and velocities, and the state elements' states.
std::vector<Real> sedan_channels(const Simulator& sim, const VehicleHandle& vh)
{
    const auto& s = sim.states()[static_cast<std::size_t>(vh.chassis_body)];
    std::vector<Real> c{s.p_WB.x(), s.p_WB.y(), s.p_WB.z(), s.v_WB.x(), s.v_WB.y(), s.v_WB.z(),
                        s.w_WB.x(), s.w_WB.y(), s.w_WB.z()};
    for (Index k = 0; k < sim.x.size(); ++k) c.push_back(sim.x(k));
    return c;
}

Record sedan_record(const Simulator& sim)
{
    Record r{"t [s]",
             {"chassis x [m]", "chassis y [m]", "chassis z [m]", "chassis vx [m/s]", "chassis vy [m/s]",
              "chassis vz [m/s]", "chassis wx [rad/s]", "chassis wy [rad/s]", "chassis wz [rad/s]"},
             {}};
    for (Index k = 0; k < sim.x.size(); ++k) r.channels.push_back("state " + std::to_string(k));
    return r;
}

/// The default sedan (simple corners: each wheel on a slider under the
/// chassis) with its Pacejka tyres and drivetrain, from its equilibrium: 0.3 s
/// at rest, 1 s at half throttle, 1 s steering 0.03 rad to the left. RK4 at
/// 1 ms. ISO 8855 axes since task 7.1.
Record sedan_drive(Real scale)
{
    const VehicleTemplate tmpl = VehicleTemplate::DefaultSedan();
    System sys;
    VehicleHandle vh = build_vehicle(sys, tmpl);
    sys.model.gravity *= scale;
    Simulator sim(sys);
    sim.method = Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    Record r = sedan_record(sim);
    r.add(0.0, sedan_channels(sim, vh));
    dt.throttle = 0.0;
    for (int k = 0; k < 46; ++k) {
        if (k == 6) dt.throttle = 0.5;
        if (k == 26) vh.set_steering(0.03);
        sim.run(0.05, 1e-3);
        r.add(sim.time, sedan_channels(sim, vh));
    }
    return r;
}

/// A slalom of the same sedan, 20 s: steering 0.04 sin(2 pi 0.5 t) at a
/// quarter throttle. The long suite's.
Record sedan_slalom(Real scale)
{
    const VehicleTemplate tmpl = VehicleTemplate::DefaultSedan();
    System sys;
    VehicleHandle vh = build_vehicle(sys, tmpl);
    sys.model.gravity *= scale;
    Simulator sim(sys);
    sim.method = Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    dt.throttle = 0.25;
    Record r = sedan_record(sim);
    for (int k = 0; k < 200; ++k) {
        vh.set_steering(0.04 * std::sin(2.0 * pi * 0.5 * sim.time));
        sim.run(0.1, 1e-3);
        r.add(sim.time, sedan_channels(sim, vh));
    }
    return r;
}

/// The chassis's placement and velocities, then for each corner (FL, FR, RL,
/// RR) its tyre's vertical and lateral forces at the last evaluation and its
/// upright's toe and camber, then the state elements' states.
std::vector<Real> car_channels(const Simulator& sim, const VehicleHandle& vh)
{
    const auto& s = sim.states()[static_cast<std::size_t>(vh.chassis_body)];
    std::vector<Real> c{s.p_WB.x(), s.p_WB.y(), s.p_WB.z(), s.v_WB.x(), s.v_WB.y(), s.v_WB.z(),
                        s.w_WB.x(), s.w_WB.y(), s.w_WB.z()};
    for (int k = 0; k < 4; ++k) {
        const RigidBodyState& w = sim.states()[static_cast<std::size_t>(vh.upright(k))];
        c.insert(c.end(), {vh.tire(k)->get_vertical_force(), vh.tire(k)->get_Fy(), extract_toe(w),
                           extract_camber(w)});
    }
    for (Index k = 0; k < sim.x.size(); ++k) c.push_back(sim.x(k));
    return c;
}

Record car_record(const Simulator& sim)
{
    Record r = sedan_record(sim);
    std::vector<std::string> names(r.channels.begin(), r.channels.begin() + 9);
    for (const char* corner : {"FL", "FR", "RL", "RR"}) {
        for (const char* q : {" Fz [N]", " Fy [N]", " toe [rad]", " camber [rad]"}) {
            names.push_back(std::string(corner) + q);
        }
    }
    names.insert(names.end(), r.channels.begin() + 9, r.channels.end());
    r.channels = names;
    return r;
}

/// The sports car (double wishbones, the front steered through its rack) with
/// anti-roll bars and downforce that grows as the ride height falls, its
/// centre of pressure off the centre line: 0.3 s at rest, 0.7 s at 0.6
/// throttle, 1 s steering 0.05 rad to the left. RK4 at 1 ms. Added before
/// task 7.1 moved the vehicle layer to ISO 8855, to compare the two (its
/// legacy centre of pressure was (0.2, 0.05, 0.03): forward, up, left).
Record sports_car_turn(Real scale)
{
    VehicleTemplate tmpl = VehicleTemplate::SportsCar();
    tmpl.front_axle.k_arb = 30000.0;
    tmpl.front_axle.c_arb = 500.0;
    tmpl.rear_axle.k_arb = 15000.0;
    tmpl.chassis.ClA = 1.2;
    tmpl.chassis.aero_h_ref = 0.6;
    tmpl.chassis.aero_dClA_dh = 3.0;
    tmpl.chassis.aero_cop_offset = Vec3(0.2, 0.03, 0.05);   // forward, left, up (ISO 8855)
    System sys;
    VehicleHandle vh = build_vehicle(sys, tmpl);
    sys.model.gravity *= scale;
    Simulator sim(sys);
    sim.method = Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    vh.install_anti_roll_bars(sys, sim.states());
    vh.install_aerodynamics(sys);
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    Record r = car_record(sim);
    for (int k = 0; k < 40; ++k) {
        if (k == 6) dt.throttle = 0.6;
        if (k == 20) vh.set_steering(0.05);
        sim.run(0.05, 1e-3);
        r.add(sim.time, car_channels(sim, vh));
    }
    return r;
}

/// The front-wheel-drive hatchback (McPherson struts in front, steered
/// through their rack; simple corners behind): 0.3 s at rest, 0.7 s at half
/// throttle, then 1 s braking at 0.3 while steering 0.04 rad to the right.
/// RK4 at 1 ms. Added before task 7.1, as above.
Record hatchback_brake(Real scale)
{
    const VehicleTemplate tmpl = VehicleTemplate::FWDHatchback();
    System sys;
    VehicleHandle vh = build_vehicle(sys, tmpl);
    sys.model.gravity *= scale;
    Simulator sim(sys);
    sim.method = Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    Record r = car_record(sim);
    for (int k = 0; k < 40; ++k) {
        if (k == 6) dt.throttle = 0.5;
        if (k == 20) {
            dt.throttle = 0.0;
            dt.brake = 0.3;
            vh.set_steering(-0.04);
        }
        sim.run(0.05, 1e-3);
        r.add(sim.time, car_channels(sim, vh));
    }
    return r;
}

/// The sedan on an ISO 8608 road of class C (seed 1, a 200 m period), from
/// its flat-road equilibrium at 15 m/s with a quarter throttle, steering
/// 0.03 rad to the left after 0.5 s; 2 s, RK4 at 1 ms. The tyres take the
/// ground from the road (task 7.2). The road is the same across its width, so
/// without the steering the car's roll, yaw and lateral motion would be zero
/// but for rounding, and so would their tolerances.
Record sedan_rough_road(Real scale)
{
    const VehicleTemplate tmpl = VehicleTemplate::DefaultSedan();
    System sys;
    VehicleHandle vh = build_vehicle(sys, tmpl);
    Iso8608Params road;
    road.road_class = Iso8608Class::C;
    road.seed = 1;
    road.length = 200.0;
    vh.set_road(std::make_shared<Iso8608Road>(road));
    sys.model.gravity *= scale;
    Simulator sim(sys);
    sim.method = Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.v(sys.model.idx_v[static_cast<std::size_t>(vh.chassis_body)] + 3) = 15.0;   // forward
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    dt.throttle = 0.25;
    Record r = car_record(sim);
    for (int k = 0; k < 40; ++k) {
        if (k == 10) vh.set_steering(0.03);
        sim.run(0.05, 1e-3);
        r.add(sim.time, car_channels(sim, vh));
    }
    return r;
}

/// The sedan's static equilibrium (to 1e-9) and its modes at rest: every
/// mode faster than 1e-3 rad/s (the seven neutral directions excluded: the
/// position on the ground, the heading, and since task 7.3 the four spin
/// angles). The wheels' spins, held by their tyres' slip at rest, decay at
/// about 3000 per second, ten times faster than the body's fastest mode:
/// they have a channel of their own, so that the body's modes keep a
/// tolerance of their own scale (each channel's is 1e-8 of its largest
/// value; in one channel with the spins, a change of 1e-6 moved the body's
/// modes by only 8.5 times it).
Record sedan_modes(Real scale)
{
    System sys;
    VehicleHandle vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());
    sys.model.gravity *= scale;
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    StaticOptions options;
    options.tolerance = 1e-9;
    (void)static_equilibrium(sim, options);
    const LinearModel lin = linearize(sim);
    Record r{"mode", {"natural frequency [rad/s]", "damping ratio", "wheel spin decay rate [1/s]"}, {}};
    int k = 0;
    for (const Mode& m : lin.modes) {
        if (m.natural_frequency <= 1e-3) continue;
        if (!m.oscillatory() && m.natural_frequency > 1000.0) {
            r.add(k++, {0.0, 0.0, m.natural_frequency});
        } else {
            r.add(k++, {m.natural_frequency, m.damping_ratio, 0.0});
        }
    }
    return r;
}

// --- Analyses ----------------------------------------------------------------

/// The sports car's front left double wishbone swept from -0.05 to 0.05 m of
/// travel; the perturbation moves the upper ball joint's hardpoint.
Record wishbone_sweep(Real scale)
{
    VehicleTemplate tmpl = VehicleTemplate::SportsCar();
    tmpl.front_axle.dwb.uca_outer_offset *= scale;
    System sys;
    auto [dwb, bump] = build_dwb_for_analysis(sys, tmpl, 0);
    (void)bump;
    Kinematics k(sys);
    const KinematicSweepResult sweep = sweep_bump_travel(k, dwb.upright_body, -0.05, 0.05, 21);
    Record r{"bump [m]", {"camber [rad]", "toe [rad]", "wheel centre height [m]", "converged"}, {}};
    for (const auto& p : sweep.points) r.add(p.bump, {p.camber, p.toe, p.wheel_height, p.converged ? 1.0 : 0.0});
    return r;
}

/// The lap speed profile of an oval (two 200 m straights, two half circles of
/// 60 m), the default lap vehicle; the perturbation scales its friction.
Record lap_oval(Real scale)
{
    Track t;
    t.add_straight(200.0);
    t.add_arc_by_angle(pi, 60.0);
    t.add_straight(200.0);
    t.add_arc_by_angle(pi, 60.0);
    LapVehicle lv;
    lv.mu *= scale;
    const LapResult lap = simulate_lap(t, lv, 400, true);
    Record r{"s [m]", {"speed [m/s]", "lap time [s]"}, {}};
    for (std::size_t k = 0; k < lap.s.size(); k += 20) r.add(lap.s[k], {lap.V[k], lap.lap_time});
    return r;
}

} // namespace

const std::vector<Scenario>& scenarios()
{
    static const std::vector<Scenario> all = {
        {"four-bar", "A planar four-bar closed by a revolute closure, from rest; RK4 at 1 ms for 2 s.", "gravity",
         1e-8, false, four_bar},
        {"heavy-top", "A heavy top on a spherical closure, spun at 30 rad/s; generalised-alpha at 1 ms for 2 s.",
         "gravity", 1e-8, false, heavy_top},
        {"bushed-pendulum", "The bushed double pendulum (1e6 N/m); Radau IIA at fixed steps of 1 ms for 1 s.",
         "gravity", 1e-8, false, bushed_pendulum},
        {"hmmwv-joints", "The HMMWV of task 5.4 over the bump (front at 0.2 s); RK4 at 0.5 ms for 1.2 s.",
         "gravity", 1e-8, false,
         [](Real s) { return hmmwv(s, false, Integrator::RK4, 5e-4, 1.2, 0.2); }},
        {"hmmwv-bushings", "The HMMWV with bushings over the bump; generalised-alpha at 1 ms for 1.2 s.",
         "gravity", 1e-8, false,
         [](Real s) { return hmmwv(s, true, Integrator::GeneralizedAlpha, 1e-3, 1.2, 0.2); }},
        {"sedan-drive", "The sedan (simple corners) at rest, accelerating, then steering; RK4 at 1 ms for 2.3 s.",
         "gravity", 1e-8, false, sedan_drive},
        {"sedan-modes", "The sedan's static equilibrium and its modes at rest.", "gravity", 1e-8, false,
         sedan_modes},
        {"sports-car-turn",
         "The double-wishbone sports car with anti-roll bars and downforce, accelerating, then turning left; "
         "RK4 at 1 ms for 2 s.",
         "gravity", 1e-8, false, sports_car_turn},
        {"sedan-rough-road",
         "The sedan (simple corners) at 15 m/s on an ISO 8608 road of class C, steering; RK4 at 1 ms for 2 s.",
         "gravity", 1e-8, false, sedan_rough_road},
        {"hatchback-brake",
         "The front-wheel-drive McPherson hatchback, accelerating, then braking in a right turn; RK4 at 1 ms "
         "for 2 s.",
         "gravity", 1e-8, false, hatchback_brake},
        {"wishbone-sweep", "The sports car's front double wishbone over +-0.05 m of travel.",
         "the upper ball joint's hardpoint", 1e-8, false, wishbone_sweep},
        {"lap-oval", "The lap speed profile of an oval of 200 m straights and 60 m turns.", "the friction",
         1e-8, false, lap_oval},
        {"hmmwv-joints-5s", "The HMMWV over the bump at 0.5 s; RK4 at 0.25 ms for the paper's 5 s.", "gravity",
         1e-8, true, [](Real s) { return hmmwv(s, false, Integrator::RK4, 2.5e-4, 5.0, 0.5); }},
        {"hmmwv-bushings-5s", "The HMMWV with bushings over the bump at 0.5 s; generalised-alpha at 0.5 ms for 5 s.",
         "gravity", 1e-8, true,
         [](Real s) { return hmmwv(s, true, Integrator::GeneralizedAlpha, 5e-4, 5.0, 0.5); }},
        {"sedan-slalom", "The sedan in a 20 s slalom at a quarter throttle; RK4 at 1 ms.", "gravity", 1e-8, true,
         sedan_slalom},
    };
    return all;
}

} // namespace mbd_regression
