// The examples of the how-to guide docs/help/howto/vehicles.md (task H.6,
// Phase 7's tasks 7.1 to 7.12), each run and checked against a value derived
// from the geometry or the physics. The guide quotes the code between the
// example markers exactly (test_howto_pages.cpp).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "mbd/forces/simple_tyres.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/vehicle/brakes.hpp"
#include "mbd/vehicle/mass_properties.hpp"
#include "mbd/vehicle/powertrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::WithinRel;
using namespace mbd;
using namespace mbd::kernel;

namespace {

/// The sedan (simple corners) on linear tyres without rolling resistance,
/// stiff vertically (2e7 N/m) and without lag: a car whose motion the hand
/// calculations below describe.
VehicleTemplate plain_sedan()
{
    VehicleTemplate t = VehicleTemplate::DefaultSedan();
    LinearTyreParams lt;
    lt.rolling_resistance = 0.0;
    lt.vertical_stiffness = 2e7;
    t.front_axle.tyre_model = std::make_shared<LinearTyre>(lt);
    t.rear_axle.tyre_model = t.front_axle.tyre_model;
    return t;
}

Real speed(const Simulator& sim, const VehicleHandle& car)
{
    const RigidBodyState& s = sim.states()[static_cast<std::size_t>(car.chassis_body)];
    return s.v_WB.dot(s.q_WB * Vec3::UnitX());
}

} // namespace

TEST_CASE("How-to vehicles: a car from its template, with a driver, and its mass report", "[howto]")
{
    // The sports car with a 75 kg driver 0.37 m left of its centre line.
    // Every other mass of the car is mirrored across that line, so the
    // report's centre of mass lies 75 x 0.37 / M to the left, M the whole
    // mass with the driver (t.total_mass()); the left corners carry more.
    // Allowed 1e-12 m (the sums' rounding) and 1e-9 kg.
    // [example: vehicle-build]
    VehicleTemplate t = VehicleTemplate::SportsCar();
    t.chassis.lumped_masses = {{"driver", 75.0, Vec3(-0.3, 0.37, 0.15), Mat3::Zero()}};
    System sys;
    VehicleHandle car = build_vehicle(sys, t);
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, car);   // at rest at its design position, on a level road
    sim.initialize();
    const MassReport report = mass_report(sim, car);
    std::cout << report;   // mass, centre of mass and its height, inertia, sprung and unsprung per corner
    // [end example]
    const Real M = t.total_mass();
    CHECK(std::abs(report.total_mass - M) < 1e-9);
    CHECK(std::abs(report.centre_of_mass.y() - 75.0 * 0.37 / M) < 1e-12);
    CHECK(report.corner_mass(0) > report.corner_mass(1));
    CHECK(report.corner_mass(2) > report.corner_mass(3));
}

TEST_CASE("How-to vehicles: a corner alone, and its camber gain", "[howto]")
{
    // The sports car's front left double wishbone on its rig. Its arms turn
    // about axes along X and its joints lie in the wheel centre's
    // front-view plane, so in that view the upright turns about the point
    // where the arms' lines meet (the instant centre, IC): its turn about X
    // per metre of the wheel centre's rise is 1 / (W_y - IC_y), the inverse
    // of the front-view swing arm (task 7.12). Allowed 1e-10 rad/m (a
    // linear solve's rounding).
    const VehicleTemplate t = VehicleTemplate::SportsCar();
    // [example: vehicle-corner]
    System rig_sys;
    const CornerRig rig = build_corner_rig(rig_sys, t, 0);   // the front left corner, the chassis fixed
    Kinematics k(rig_sys);
    k.solve();                          // the design position
    const VecX v = k.velocities();      // per metre of the wheel centre's rise (its driver's rate)
    Data data(rig_sys.model);
    forward_kinematics(rig_sys.model, data, k.q, v);
    const Real camber_gain = body_state(data, rig.corners[0].upright_body).w_WB.x();   // [rad/m]
    // [end example]
    const DwbHardpoints& h = t.front_axle.dwb;
    const Vec3 W(t.front_axle_x, t.front_axle.half_track, 0.0);
    const Vec3 a = W + h.lca_pivot_offset, b = W + h.lca_outer_offset;
    const Vec3 c = W + h.uca_pivot_offset, d = W + h.uca_outer_offset;
    Eigen::Matrix2d A;
    A << b.y() - a.y(), -(d.y() - c.y()),
         b.z() - a.z(), -(d.z() - c.z());
    const Eigen::Vector2d st = A.colPivHouseholderQr().solve(Eigen::Vector2d(c.y() - a.y(), c.z() - a.z()));
    const Real ic_y = a.y() + st(0) * (b.y() - a.y());
    INFO("camber gain " << camber_gain << " rad/m; 1 / swing arm " << 1.0 / (W.y() - ic_y));
    CHECK(std::abs(camber_gain - 1.0 / (W.y() - ic_y)) < 1e-10);
}

TEST_CASE("How-to vehicles: brakes, and the stopping distance", "[howto]")
{
    // The plain sedan (1560 kg, four 40 kg wheels, I = m R^2 / 2 each) from 20
    // m/s, the pedal at half (6 MPa), the pressure rising as 1 - e^(-t /
    // tau). No wheel locks, so the deceleration is d(t) = d_inf (1 -
    // e^(-t / tau)), d_inf = 2 (1.9e-4 + 0.9e-4) p / R / (m + 4 I / R^2),
    // and v = 0 where v0 = d_inf (t - tau (1 - e^(-t / tau))), the distance
    // s = v0 t - d_inf (t^2 / 2 - tau t + tau^2 (1 - e^(-t / tau))). As in
    // task 7.9's test: allowed 3e-3, sized to the last metre per second, where
    // the tyres' force fades (V_low^2 / (2 d) = 0.077 m of the 31 m).
    const VehicleTemplate t = plain_sedan();
    System sys;
    VehicleHandle car = build_vehicle(sys, t);
    for (int c = 0; c < 4; ++c) car.tire(c)->lag = false;
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, car);
    sim.v(sys.model.idx_v[static_cast<std::size_t>(car.chassis_body)] + 3) = 20.0;
    sim.refresh();
    roll_wheels(sim, car);
    sim.initialize();
    const Real x0 = sim.states()[static_cast<std::size_t>(car.chassis_body)].p_WB.x();
    // [example: vehicle-brakes]
    BrakeSystem brakes(BrakeSystemParams{});   // 1.9e-4 and 0.9e-4 N m/Pa, 12 MPa at full pedal
    brakes.connect(sim, car);
    brakes.pedal = 0.5;
    while (speed(sim, car) > 0.01) sim.step(5e-4);
    // [end example]
    const Real distance = sim.states()[static_cast<std::size_t>(car.chassis_body)].p_WB.x() - x0;
    const BrakeSystemParams bp;
    const Real R = t.front_axle.tyre_body().free_radius, m = t.total_mass(), I = 0.5 * 40.0 * R * R;
    const Real d_inf = 2.0 * (bp.front_gain + bp.rear_gain) * 0.5 * bp.max_pressure / R / (m + 4.0 * I / (R * R));
    const Real tau = bp.response_time, v0 = 20.0;
    Real stop = v0 / d_inf;
    for (int i = 0; i < 50; ++i) {
        const Real v = v0 - d_inf * (stop - tau * (1.0 - std::exp(-stop / tau)));
        stop += v / (d_inf * (1.0 - std::exp(-stop / tau)));
    }
    const Real s = v0 * stop - d_inf * (0.5 * stop * stop - tau * stop + tau * tau * (1.0 - std::exp(-stop / tau)));
    INFO("stopped in " << distance << " m and " << sim.time << " s; by hand " << s << " m and " << stop << " s");
    CHECK_THAT(distance, WithinRel(s, 3e-3));
}

TEST_CASE("How-to vehicles: a powertrain, and the acceleration it gives", "[howto]")
{
    // The plain sedan driven at the rear by a motor of 100 N m through a ratio
    // of 8 at an efficiency of 0.92: the axle's torque 736 N m, the force at
    // the road 736 / R. The car, its wheels and the driveline accelerate
    // together once the half-shafts' ringing has died (exp(-4.4 t), task
    // 7.12): a = (736 / R) / M_eff, M_eff = m + (4 I_w + J_c + 2 J_g +
    // (J_in + J_m) 8^2) / R^2, the carrier and side gears turning with the
    // wheels and the gearbox's input and the motor 8 times faster. Measured
    // between 2.5 and 3 s. Left out: the tyres' loaded radius, R less 1.9e-4
    // m (6e-4 of R), in the force's lever and in the wheels' speed. Allowed
    // 2e-3.
    const VehicleTemplate t = plain_sedan();
    System sys;
    VehicleHandle car = build_vehicle(sys, t);
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, car);
    sim.initialize();
    // [example: vehicle-powertrain]
    PowertrainParams p;
    p.coupling = Coupling::Direct;                        // a motor on the gearbox's input
    p.engine = EngineMap::motor(100.0, 1e6, 1e6, 0.02);   // 100 N m at any speed here, 0.02 kg m^2
    p.gear_ratios = {1.0};
    p.final_drive = 8.0;
    p.automatic = false;
    Powertrain powertrain(p);   // rear-wheel drive, an open differential, compliant half-shafts
    powertrain.connect(sim, car);
    powertrain.throttle = 1.0;
    sim.run(2.5, 5e-4);
    // [end example]
    const Real v1 = speed(sim, car);
    sim.run(0.5, 5e-4);
    const Real a = (speed(sim, car) - v1) / 0.5;
    const Real R = t.rear_axle.tyre_body().free_radius, I_w = 0.5 * 40.0 * R * R;
    const Real J = 4.0 * I_w + p.carrier_inertia + 2.0 * p.side_gear_inertia
                 + (p.input_inertia + p.engine.inertia) * 64.0;
    const Real expected = p.efficiency * 8.0 * 100.0 / R / (t.total_mass() + J / (R * R));
    INFO("acceleration " << a << " m/s^2; by hand " << expected);
    CHECK_THAT(a, WithinRel(expected, 2e-3));
}

TEST_CASE("How-to vehicles: an aerodynamic map over yaw", "[howto]")
{
    // A map over yaw alone: straight ahead and at 0.2 rad, its mirror image
    // at -0.2 (the side force, roll and yaw moments change sign). The chassis
    // moving at 30 m/s with its velocity 0.1 rad to the left of its X axis
    // sees the air at a yaw of 0.1 rad, halfway between two nodes: its force
    // is q times the mean of their A's, q = rho V^2 / 2, in the chassis's
    // axes (here the world's). Allowed 1e-12 of it.
    VehicleTemplate t = VehicleTemplate::SportsCar();
    // [example: vehicle-aero-map]
    auto map = std::make_shared<AeroMap>();
    map->nodes[4] = {-0.2, 0.0, 0.2};   // yaw [rad]; the ride heights, pitch and roll keep one node
    Vec6 straight, yawed;               // A_x, A_y, A_z [m^2] and B_x, B_y, B_z [m^3], chassis axes
    straight << -0.70, 0.0, -1.20, 0.0, -0.30, 0.0;
    yawed << -0.75, 0.60, -1.00, 0.10, -0.25, 0.20;
    Vec6 mirrored = yawed;
    mirrored(1) = -yawed(1);
    mirrored(3) = -yawed(3);
    mirrored(5) = -yawed(5);
    map->values = {mirrored, straight, yawed};
    t.chassis.aero_map = map;
    System sys;
    VehicleHandle car = build_vehicle(sys, t);
    AerodynamicForce* aero = car.install_aerodynamics(sys);
    // [end example]
    Simulator sim(sys);
    const std::size_t iv = static_cast<std::size_t>(sys.model.idx_v[static_cast<std::size_t>(car.chassis_body)]);
    sim.v(static_cast<Eigen::Index>(iv) + 3) = 30.0 * std::cos(0.1);
    sim.v(static_cast<Eigen::Index>(iv) + 4) = 30.0 * std::sin(0.1);
    sim.refresh();
    std::vector<RigidBodyForces> f(sim.states().size());
    aero->apply(sim.states(), f);
    const Real q = 0.5 * aero->params.air_density * 900.0;
    const Vec3 expected = q * 0.5 * (straight.head<3>() + yawed.head<3>());
    const Vec3 F = f[static_cast<std::size_t>(car.chassis_body)].f_W;
    INFO("force " << F.transpose() << " N; by hand " << expected.transpose());
    CHECK((F - expected).cwiseAbs().maxCoeff() < 1e-12 * expected.norm());
}
