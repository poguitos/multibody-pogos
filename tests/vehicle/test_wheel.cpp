// The wheel as a body on its spin joint (plan task 7.3, D62):
//   - its inertia, a uniform thin disc;
//   - a spinning wheel on a pivot precesses at m g l / (I_s omega_s), the
//     plan's "done when" (second part);
//   - roll_wheels starts every tyre without longitudinal slip, the chassis
//     pitching or not;
//   - the drivetrain's torques come in equal and opposite pairs: drive between
//     wheel and chassis, brake between wheel and upright.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/drivetrain.hpp"
#include "mbd/vehicle/vehicle.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/wheel.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

TEST_CASE("Wheel: its inertia is a uniform thin disc about its Y axis", "[vehicle][wheel]")
{
    // m = 40 kg, R = 0.35 m: m R^2 / 2 = 2.45 kg m^2 about the axle (the
    // drivetrain's value before task 7.3), m R^2 / 4 = 1.225 about a diameter.
    const RigidBodyInertia I = wheel_inertia(40.0, 0.35);
    CHECK(I.mass == 40.0);
    CHECK(I.com_B.isZero());
    CHECK_THAT(I.I_com_B(1, 1), WithinAbs(2.45, 1e-12));
    CHECK_THAT(I.I_com_B(0, 0), WithinAbs(1.225, 1e-12));
    CHECK_THAT(I.I_com_B(2, 2), WithinAbs(1.225, 1e-12));
    CHECK(I.I_com_B(0, 1) == 0.0);
    CHECK(I.I_com_B(0, 2) == 0.0);
    CHECK(I.I_com_B(1, 2) == 0.0);
}

namespace
{
    struct Gyroscope {
        Real precession_angle;   ///< About +Z, from the axle's horizontal direction [rad]
        Real elevation;          ///< The axle's Z component (zero if it stays horizontal)
        Real spin;               ///< The spin joint's rate at the end [rad/s]
    };

    /// A wheel (m = 10 kg, R = 0.3 m) on its spin joint, its centre l = 0.2 m
    /// along the Y axis of a carrier on a spherical joint at the origin (a
    /// small ball at the pivot: 0.5 kg, 0.01 kg m^2 about every axis, so that
    /// the carrier's turning about the axle has an inertia of its own). Axle
    /// horizontal, the wheel spinning at `spin`, the carrier turning about Z
    /// at `precession`; run for T with RK4 at h.
    Gyroscope gyroscope(Real spin, Real precession, Real T, Real h)
    {
        kernel::System sys;
        sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
        const BodyIndex carrier = sys.model.add_body(
            0, std::make_shared<kernel::SphericalJointModel>(), Transform3::Identity(), Transform3::Identity(),
            RigidBodyInertia(0.5, Vec3::Zero(), 0.01 * Mat3::Identity()), "carrier");
        const BodyIndex wheel = add_wheel(sys.model, carrier, Vec3(0.0, 0.2, 0.0), 10.0, 0.3, "wheel");
        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        // The spherical joint's velocity is the carrier's angular velocity in
        // its own axes, which are the world's at the start.
        sim.v.segment<3>(sys.model.idx_v[carrier]) = Vec3(0.0, 0.0, precession);
        sim.v(sys.model.idx_v[wheel]) = spin;
        sim.initialize();
        sim.run(T, h);
        const RigidBodyState& w = sim.states()[static_cast<std::size_t>(wheel)];
        const Vec3 axle = w.q_WB * Vec3::UnitY();
        return {std::atan2(-axle.x(), axle.y()), axle.z(), sim.v(sys.model.idx_v[wheel])};
    }
}

TEST_CASE("Wheel: a spinning wheel on a pivot precesses at m g l / (I_s omega_s)", "[vehicle][wheel][gyroscope]")
{
    // The plan's "done when" for task 7.3, second part. A symmetric top with
    // its axle horizontal precesses steadily when
    //     m g l = Omega_p (I_s omega_3 - I_1 Omega_p cos(theta)),
    // theta the axle's angle from the vertical (Goldstein, the heavy
    // symmetric top). At theta = 90 deg the second term vanishes, whatever
    // the transverse inertia I_1: m g l = I_s omega_s Omega_p exactly, with
    // omega_s the spin about the axle, here the spin joint's rate (the
    // precession, about Z, has no part along the horizontal axle). The ball
    // at the pivot adds no moment of gravity and, being the same about every
    // axis, nothing to the balance of moments. Gravity's moment about the
    // pivot is (0, l, 0) x (0, 0, -m g) = (-m g l, 0, 0); the spin's
    // angular momentum I_s omega_s Y turns about +Z at Omega_p to follow it:
    // Omega_p Z x I_s omega_s Y = -Omega_p I_s omega_s X.
    //
    // m = 10 kg, R = 0.3 m: I_s = m R^2 / 2 = 0.45 kg m^2; l = 0.2 m,
    // omega_s = 100 rad/s: Omega_p = 10 x 9.80665 x 0.2 / 45 = 0.43585 rad/s.
    // Started exactly so, the axle stays horizontal and turns at Omega_p: in
    // 3 s by 1.3076 rad. Spun the other way it precesses the other way.
    // Without the wheel's spin in the multibody model (before task 7.3) the
    // wheel would simply fall.
    //
    // In the steady precession every joint's acceleration is zero (the
    // spherical joint's rate, in the carrier's own axes, and the spin are
    // constant), so RK4's four stages agree and each step is exact but for
    // rounding: measured 9e-15 rad after 3 s at 1 ms, 1.8e-14 at 0.5 ms
    // (rounding, growing with the steps), the elevation 3e-16. Allowed
    // 1e-12 rad. Anything that unbalances the moments starts a nutation (at
    // I_s omega_s / I_1 = 45 / (0.225 + 0.4 + 0.01) = 71 rad/s): started
    // with a precession 1e-6 too fast, the axle ended 3.0e-9 rad above the
    // horizontal and 5.2e-9 rad behind (measured; 1e-4 too fast, a hundred
    // times more), far outside these tolerances.
    const Real m = 10.0, R = 0.3, l = 0.2, omega_s = 100.0, T = 3.0;
    const Real I_s = 0.5 * m * R * R;
    const Real Omega_p = m * g_accel * l / (I_s * omega_s);

    const Gyroscope run = gyroscope(omega_s, Omega_p, T, 1e-3);
    const Gyroscope fine = gyroscope(omega_s, Omega_p, T, 5e-4);
    const Gyroscope reversed = gyroscope(-omega_s, -Omega_p, T, 1e-3);
    INFO("precession error " << run.precession_angle - Omega_p * T << " rad of " << Omega_p * T
                             << ", at half the step " << fine.precession_angle - Omega_p * T << "; elevation "
                             << run.elevation << ", at half the step " << fine.elevation << "; reversed: error "
                             << reversed.precession_angle + Omega_p * T << ", elevation " << reversed.elevation
                             << "; spin error at the end " << run.spin - omega_s);
    CHECK_THAT(run.precession_angle, WithinAbs(Omega_p * T, 1e-12));
    CHECK_THAT(run.elevation, WithinAbs(0.0, 1e-12));
    CHECK_THAT(fine.precession_angle, WithinAbs(Omega_p * T, 1e-12));
    CHECK_THAT(reversed.precession_angle, WithinAbs(-Omega_p * T, 1e-12));
    CHECK_THAT(reversed.elevation, WithinAbs(0.0, 1e-12));
    // The spin about the axle is constant: no moment acts along it.
    CHECK_THAT(run.spin, WithinAbs(omega_s, 1e-12));
}

TEST_CASE("Wheel: roll_wheels starts every tyre without longitudinal slip", "[vehicle][wheel]")
{
    // The simple vehicle at its equilibrium, given 10 m/s forward and, to
    // make the spin joints' rates differ from the wheels' spins, a pitch rate
    // of 0.3 rad/s: each wheel's spin about its axle is its joint's rate plus
    // the carrier's (the chassis's) 0.3. roll_wheels sets each joint so that
    // the spin is the tyre's free rolling: the longitudinal slip speed Vx -
    // Omega R_eff (task 7.4: the tyre's slip ratio is its deflection's, zero
    // at the start whatever the spin) is then zero but for rounding, 1e-12 of
    // 10 m/s. Forward rolling is a positive spin: the joint's rate is near (10
    // / 0.304) - 0.3 = 32.6 rad/s (the Appendix 3 tyre's R_eff, 0.3135 m less
    // half its static deflection).
    kernel::System sys;
    VehicleParams vp;
    VehicleModel vm = build_simple_vehicle(sys, vp);
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vm);
    const int cv = sys.model.idx_v[static_cast<std::size_t>(vm.chassis_body)];
    sim.v(cv + 1) = 0.3;    // pitch rate, chassis axes
    sim.v(cv + 3) = 10.0;   // forward, chassis axes
    sim.initialize();
    roll_wheels(sim, vm);
    sim.acceleration(sim.q, sim.v, sim.time);   // the tyres' forces at this state
    for (std::size_t c = 0; c < 4; ++c) {
        INFO("corner " << c << ": slip speed " << vm.tires[c]->last_Vsx << " m/s, joint rate "
                       << sim.v(vm.spin_v[c]));
        CHECK_THAT(vm.tires[c]->last_Vsx, WithinAbs(0.0, 1e-11));
        CHECK(sim.v(vm.spin_v[c]) > 30.0);
        CHECK(sim.v(vm.spin_v[c]) < 35.0);
    }
}

TEST_CASE("Wheel: the drivetrain's torques act in pairs, drive against the chassis, brake against the upright",
          "[vehicle][wheel][drivetrain]")
{
    // The sports car (double wishbones) at 15 m/s, a little throttle and
    // brake together, after 0.2 s. The drivetrain's torques on all bodies:
    // each wheel takes its drive and brake torques about its axle; the
    // chassis takes the drive torques back (an inboard differential); each
    // upright takes its brake's back (an outboard brake). No force, and the
    // moments add up to zero (rounding of 1e-9 of the largest), so the
    // drivetrain cannot change the car's momentum. A wheel rolling forward,
    // not locked, is braked by its whole brake torque against the roll.
    VehicleTemplate tmpl = VehicleTemplate::SportsCar();
    kernel::System sys;
    VehicleHandle vh = build_vehicle(sys, tmpl);
    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.v(sys.model.idx_v[static_cast<std::size_t>(vh.chassis_body)] + 3) = 15.0;
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    dt.throttle = 0.4;
    dt.brake = 0.1;
    sim.run(0.2, 1e-3);

    std::vector<RigidBodyForces> forces(static_cast<std::size_t>(sys.model.nbodies()));
    dt.apply_torques(sim.states(), forces);
    Vec3 force_sum = Vec3::Zero(), moment_sum = Vec3::Zero();
    Real largest = 0.0;
    for (const RigidBodyForces& f : forces) {
        force_sum += f.f_W;
        moment_sum += f.tau_W;   // pure torques: no forces, so no moment arms
        largest = std::max(largest, f.tau_W.norm());
    }
    INFO("largest torque " << largest << " N m; sum " << moment_sum.transpose());
    CHECK(force_sum.isZero());
    CHECK(moment_sum.norm() <= 1e-9 * largest);
    CHECK(largest > 100.0);

    Vec3 chassis_expected = Vec3::Zero();
    for (int c = 0; c < 4; ++c) {
        INFO("corner " << c);
        const auto wheel = static_cast<std::size_t>(vh.wheel(c));
        const Vec3 axle = sim.states()[wheel].q_WB * Vec3::UnitY();
        CHECK(!dt.wheel_locked[c]);
        CHECK(dt.wheel_omega[c] > 0.0);
        CHECK(dt.brake_torque_applied[c] == -dt.brake_torque_out[c]);
        const Vec3 on_wheel = (dt.drive_torque[c] + dt.brake_torque_applied[c]) * axle;
        CHECK((forces[wheel].tau_W - on_wheel).norm() <= 1e-12 * largest);
        const auto upright = static_cast<std::size_t>(vh.upright(c));
        CHECK((forces[upright].tau_W + dt.brake_torque_applied[c] * axle).norm() <= 1e-12 * largest);
        chassis_expected -= dt.drive_torque[c] * axle;
    }
    const auto chassis = static_cast<std::size_t>(vh.chassis_body);
    CHECK((forces[chassis].tau_W - chassis_expected).norm() <= 1e-12 * largest);
    CHECK(dt.drive_torque[2] > 0.0);   // rear-wheel drive
    CHECK(dt.drive_torque[0] == 0.0);
}
