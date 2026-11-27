// Task 7.12: an analytic check for every subsystem. Most of Phase 7's
// subsystems were checked against closed forms by their own tasks (the
// table in docs/validation.md says where); these cases close the gaps left:
// the suspensions' camber gain and wheel-centre path against the front-view
// instant centre, the five-link against the double wishbone it can describe,
// and the brakes holding against the powertrain through their drive-torque
// hook.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <memory>

#include <Eigen/Dense>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/vehicle/brakes.hpp"
#include "mbd/vehicle/powertrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::WithinRel;
using namespace mbd;

namespace {

/// The upright's angular velocity and its wheel centre's velocity per unit
/// rise of the wheel centre, at the design position: corner `corner` of `t`
/// alone on its rig, from the mechanism's velocities (Kinematics).
struct Rates {
    Vec3 w;
    Vec3 v;
};

Rates design_rates(const VehicleTemplate& t, int corner)
{
    kernel::System sys;
    const CornerRig rig = build_corner_rig(sys, t, corner);
    Kinematics k(sys);
    REQUIRE(k.solve(50, 1e-14));
    const VecX v = k.velocities();
    kernel::Data data(sys.model);
    kernel::forward_kinematics(sys.model, data, k.q, v);
    const CornerHandle& c = rig.corners[0];
    const RigidBodyState u = kernel::body_state(data, c.upright_body);
    return {u.w_WB, u.v_WB + u.w_WB.cross(u.q_WB * c.wheel_center_in_upright)};
}

/// Where two lines of the front view (Y, Z) meet: the line through a and b,
/// and the line through c along d.
Eigen::Vector2d meet(const Vec3& a, const Vec3& b, const Vec3& c, const Eigen::Vector2d& d)
{
    const Eigen::Vector2d A(a.y(), a.z()), B(b.y(), b.z()), C(c.y(), c.z());
    Eigen::Matrix2d M;
    M.col(0) = B - A;
    M.col(1) = -d;
    const Eigen::Vector2d st = M.colPivHouseholderQr().solve(C - A);
    return A + st(0) * (B - A);
}

/// The front-view instant centre's predictions, per unit rise of the wheel
/// centre W, for an upright turning in the front view about the point IC
/// (and about an axis in that view, which turns nothing about X): its turn
/// about X is 1 / (W_y - IC_y), the inverse of the front-view swing arm; the
/// wheel centre moves sideways by -(W_z - IC_z) / (W_y - IC_y).
struct InstantCentre {
    Real camber_rate;
    Real lateral_rate;
};

InstantCentre predicted(const Vec3& W, const Eigen::Vector2d& ic)
{
    const Real arm = W.y() - ic(0);
    return {1.0 / arm, -(W.z() - ic(1)) / arm};
}

} // namespace

// A double wishbone whose arms turn about axes along X, its ball joints in
// the wheel centre's front-view plane (no caster): the template's default.
// In that view the arms and the upright are a four-bar, so the upright turns
// about the instant centre where the two arms' lines meet. The tie rod turns
// it about the kingpin, a line in that view, which turns nothing about X.
// So the camber's rate is the inverse of the front-view swing arm. The
// velocities come from a linear solve of the constraints' Jacobian, exact
// but for rounding amplified by its conditioning: allowed 1e-10 per metre
// of travel.
TEST_CASE("Subsystems: a double wishbone turns its upright about the front-view instant centre",
          "[vehicle][subsystems]")
{
    const VehicleTemplate t = VehicleTemplate::SportsCar();
    const DwbHardpoints& h = t.front_axle.dwb;
    REQUIRE(h.arm_axis == Vec3::UnitX());
    const Vec3 W(t.front_axle_x, t.front_axle.half_track, 0.0);
    const Vec3 lower_pivot = W + h.lca_pivot_offset, lower_joint = W + h.lca_outer_offset;
    const Vec3 upper_pivot = W + h.uca_pivot_offset, upper_joint = W + h.uca_outer_offset;
    const Vec3 d_upper = upper_joint - upper_pivot;
    const Eigen::Vector2d ic = meet(lower_pivot, lower_joint, upper_pivot, Eigen::Vector2d(d_upper.y(), d_upper.z()));
    const InstantCentre p = predicted(W, ic);

    const Rates r = design_rates(t, 0);
    INFO("instant centre (y, z) " << ic.transpose() << " m; camber rate " << r.w.x() << " rad/m, predicted "
                                  << p.camber_rate << " (difference " << r.w.x() - p.camber_rate
                                  << "); wheel centre sideways " << r.v.y() << ", predicted " << p.lateral_rate
                                  << " (difference " << r.v.y() - p.lateral_rate << ")");
    CHECK(std::abs(r.v.z() - 1.0) < 1e-12);
    CHECK(std::abs(r.w.x() - p.camber_rate) < 1e-10);
    CHECK(std::abs(r.v.y() - p.lateral_rate) < 1e-10);
}

// A McPherson strut: the lower arm about an axis along X, the strut's top
// mount a chassis point kept on a line of the upright (the strut's axis),
// all in the wheel centre's front-view plane (the template's default). The
// upright's point at the top mount moves along the strut, so the instant
// centre is where the lower arm's line meets the perpendicular to the strut
// through the top mount. Allowed as for the double wishbone.
TEST_CASE("Subsystems: a McPherson strut turns its upright about the front-view instant centre",
          "[vehicle][subsystems]")
{
    const VehicleTemplate t = VehicleTemplate::FWDHatchback();
    REQUIRE(t.front_axle.suspension_type == SuspensionType::McPherson);
    const McPhersonHardpoints& h = t.front_axle.mcpherson;
    REQUIRE(h.arm_axis == Vec3::UnitX());
    const Vec3 W(t.front_axle_x, t.front_axle.half_track, 0.0);
    const Vec3 pivot = W + h.lca_pivot_offset, joint = W + h.lca_outer_offset;
    const Vec3 top = W + h.strut_top_offset, strut = top - (W + h.strut_lower_offset);
    const Eigen::Vector2d ic = meet(pivot, joint, top, Eigen::Vector2d(-strut.z(), strut.y()));
    const InstantCentre p = predicted(W, ic);

    const Rates r = design_rates(t, 0);
    INFO("instant centre (y, z) " << ic.transpose() << " m; camber rate " << r.w.x() << " rad/m, predicted "
                                  << p.camber_rate << " (difference " << r.w.x() - p.camber_rate
                                  << "); wheel centre sideways " << r.v.y() << ", predicted " << p.lateral_rate
                                  << " (difference " << r.v.y() - p.lateral_rate << ")");
    CHECK(std::abs(r.v.z() - 1.0) < 1e-12);
    CHECK(std::abs(r.w.x() - p.camber_rate) < 1e-10);
    CHECK(std::abs(r.v.y() - p.lateral_rate) < 1e-10);
}

// A five-link whose first two links run from two points of the double
// wishbone's lower pivot axis to its lower ball joint, the next two likewise
// to the upper ball joint, and the fifth along its tie rod: two links meeting
// at a point hold it on a circle about the line through their inner points,
// as an arm turning about that line does. Its motion is therefore the double
// wishbone's: the upright's turn and the wheel centre's path, every
// component, equal to rounding (allowed 1e-12 per metre of travel).
TEST_CASE("Subsystems: a five-link laid out as a double wishbone moves as the double wishbone",
          "[vehicle][subsystems]")
{
    const VehicleTemplate dwb = VehicleTemplate::SportsCar();
    const DwbHardpoints& h = dwb.front_axle.dwb;
    VehicleTemplate five = dwb;
    five.front_axle.suspension_type = SuspensionType::FiveLink;
    const Vec3 e(0.15, 0.0, 0.0);   // half the spacing of each arm's two chassis points, along its axis
    five.front_axle.five_link.inner_offset = {{h.lca_pivot_offset - e, h.lca_pivot_offset + e, h.uca_pivot_offset - e,
                                               h.uca_pivot_offset + e, h.tierod_inner_offset}};
    five.front_axle.five_link.outer_offset = {{h.lca_outer_offset, h.lca_outer_offset, h.uca_outer_offset,
                                               h.uca_outer_offset, h.tierod_outer_offset}};

    const Rates a = design_rates(dwb, 0);
    const Rates b = design_rates(five, 0);
    INFO("double wishbone: turn " << a.w.transpose() << " rad/m, wheel centre " << a.v.transpose()
                                  << "; five-link: differences " << (b.w - a.w).transpose() << " and "
                                  << (b.v - a.v).transpose());
    CHECK((b.w - a.w).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((b.v - a.v).cwiseAbs().maxCoeff() < 1e-12);
}

// The brakes and the powertrain together, through the brakes' drive_torque
// hook (task 7.9 left it untested). A motor of 100 N m coupled directly,
// through a ratio of 8 at an efficiency of 0.92, to an open rear
// differential: at rest each rear half-shaft carries 0.92 x 8 x 100 / 2 =
// 368 N m. The rear brakes alone (front_gain zero, a parking brake). Dry
// friction: the brake holds the wheel while the torque on it is within the
// brake's, and lets it turn when the torque exceeds it. So:
//   - with the brakes' torque at 1.2 times 368 N m, the car stays where it
//     is, each rear brake carrying the shaft's 368 N m;
//   - with it at 0.8 times, the rear wheels turn, the car drives off, and
//     each rear brake gives its whole torque, 294.4 N m.
// The driveline rings: the carrier with the inertia reflected through the
// ratio, (0.05 + 0.02) x 8^2 + 0.05 + 2 x 0.005 = 4.54 kg m^2, on the two
// shafts (2 x 1e4 N m/rad, 2 x 20 N m s/rad) at 66 rad/s, damping ratio
// 0.066, decaying as exp(-4.4 t). A step of the throttle would carry it to
// 1 + exp(-pi 0.066 / sqrt(1 - 0.066^2)) = 1.8 times the static torque, past
// the brake; it is raised over 0.5
// s instead, which leaves a ringing of about 17 N m (undamped estimate).
// Allowances: that ringing after 4 s at rest, exp(-17.6), 1e-9 of the
// torque; the tyres' moment about the axle at rest, which the locked brake
// also carries, under 1e-3 of it; the brake's pressure after 1 s, exp(-20)
// from its target.
TEST_CASE("Subsystems: the brakes hold the car against the powertrain while the drive torque is below theirs",
          "[vehicle][subsystems][brakes][powertrain]")
{
    const VehicleTemplate t = VehicleTemplate::DefaultSedan();
    kernel::System sys;
    VehicleHandle vh = build_vehicle(sys, t);
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();

    PowertrainParams pp;
    pp.coupling = Coupling::Direct;
    pp.engine = EngineMap::motor(100.0, 1e6, 1e6, 0.02);
    pp.gear_ratios = {1.0};
    pp.final_drive = 8.0;
    pp.automatic = false;
    pp.layout = DriveLayout::RWD;
    Powertrain pt(pp);
    pt.connect(sim, vh);

    BrakeSystemParams bp;
    bp.front_gain = 0.0;
    BrakeSystem brakes(bp);
    brakes.connect(sim, vh);
    brakes.drive_torque = [&pt](int c) { return pt.shaft_torque[static_cast<std::size_t>(c)]; };

    const Real shaft = pp.efficiency * 8.0 * 100.0 / 2.0;
    auto pedal_for = [&](Real torque) { return torque / (bp.rear_gain * bp.max_pressure); };
    auto chassis_x = [&] { return sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.x(); };
    const Real h = 5e-4;

    // Held: the brakes on for ten response times, then the throttle raised.
    brakes.pedal = pedal_for(1.2 * shaft);
    sim.run(0.5, h);
    const Real x0 = chassis_x();
    for (int i = 1; i <= 50; ++i) {
        pt.throttle = i / 50.0;
        sim.run(0.01, h);
    }
    sim.run(4.0, h);
    const Real moved = chassis_x() - x0;
    INFO("held: moved " << moved << " m; shafts " << pt.shaft_torque[2] << ", " << pt.shaft_torque[3]
                        << " N m (by hand " << shaft << "); brakes " << brakes.torque_applied[2] << ", "
                        << brakes.torque_applied[3] << " N m; tyres' moments " << vh.tire(2)->last_axle_moment
                        << ", " << vh.tire(3)->last_axle_moment << " N m");
    CHECK(brakes.locked[2]);
    CHECK(brakes.locked[3]);
    CHECK(std::abs(moved) < 1e-3);
    CHECK_THAT(pt.shaft_torque[2], WithinRel(shaft, 1e-6));
    CHECK_THAT(pt.shaft_torque[3], WithinRel(shaft, 1e-6));
    CHECK_THAT(std::abs(brakes.torque_applied[2]), WithinRel(shaft, 1e-3));
    CHECK_THAT(std::abs(brakes.torque_applied[3]), WithinRel(shaft, 1e-3));

    // Released: the brakes' torque below the shafts'.
    brakes.pedal = pedal_for(0.8 * shaft);
    const Real x1 = chassis_x();
    sim.run(1.0, h);
    const Real driven = chassis_x() - x1;
    INFO("released: moved " << driven << " m; brakes " << brakes.torque_applied[2] << ", "
                            << brakes.torque_applied[3] << " N m (by hand " << 0.8 * shaft << ")");
    CHECK_FALSE(brakes.locked[2]);
    CHECK_FALSE(brakes.locked[3]);
    CHECK(driven > 0.01);
    CHECK_THAT(std::abs(brakes.torque_applied[2]), WithinRel(0.8 * shaft, 1e-6));
    CHECK_THAT(std::abs(brakes.torque_applied[3]), WithinRel(0.8 * shaft, 1e-6));
}
