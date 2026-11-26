// Task 7.6, part 2 (D67): the suspensions. Double wishbone (with a coil-over
// or a pushrod and rocker), McPherson, five-link, trailing arm and solid
// axle, each built from the kernel's joints and constraints with its parts'
// masses where their joints put them; spring units with damper curves, stops
// and top mounts; an anti-roll bar through drop links. The plan's test: the
// motion ratio from the geometry (the spring unit's rate of extension along
// the motion, from the mechanism's velocities) equals the one from virtual
// work (the force a bump driver needs to hold the corner against a known
// spring force, from the constraint multipliers). Other expected values are
// closed forms written out here.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/validate.hpp"
#include "mbd/vehicle/axes.hpp"
#include "mbd/vehicle/drivetrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace {

/// The sedan with the given suspension on both axles (or the rear only).
VehicleTemplate car(SuspensionType front, SuspensionType rear)
{
    VehicleTemplate t = VehicleTemplate::DefaultSedan();
    t.front_axle.suspension_type = front;
    t.rear_axle.suspension_type = rear;
    return t;
}

struct Layout {
    std::string name;
    VehicleTemplate tmpl;
    int corner;
};

/// Every architecture and layout, at a front corner when it can steer, a
/// rear one otherwise.
std::vector<Layout> layouts()
{
    std::vector<Layout> out;
    out.push_back({"double wishbone, coil-over", car(SuspensionType::DoubleWishbone, SuspensionType::Simple), 0});
    VehicleTemplate pushrod = car(SuspensionType::DoubleWishbone, SuspensionType::Simple);
    pushrod.front_axle.spring_layout = SpringLayout::Pushrod;
    out.push_back({"double wishbone, pushrod", pushrod, 1});
    out.push_back({"McPherson", car(SuspensionType::McPherson, SuspensionType::Simple), 0});
    out.push_back({"five-link", car(SuspensionType::Simple, SuspensionType::FiveLink), 2});
    out.push_back({"trailing arm", car(SuspensionType::Simple, SuspensionType::TrailingArm), 3});
    out.push_back({"solid axle", car(SuspensionType::Simple, SuspensionType::SolidAxle), 2});
    return out;
}

/// A point's position and velocity from kinematics at (q, v).
struct PointMotion {
    Vec3 p, v;
};

PointMotion point_motion(const kernel::Data& data, const kernel::Marker& m)
{
    const RigidBodyState s = kernel::body_state(data, m.body);
    const Vec3 r = s.q_WB * m.X_BM.p;
    return {s.p_WB + r, s.v_WB + s.w_WB.cross(r)};
}

/// The rate of extension of the segment between two markers.
Real extension_rate(const kernel::Data& data, const kernel::Marker& a, const kernel::Marker& b)
{
    const PointMotion pa = point_motion(data, a), pb = point_motion(data, b);
    const Vec3 e = (pb.p - pa.p).normalized();
    return (pb.v - pa.v).dot(e);
}

Real largest_acceleration(kernel::Simulator& sim)
{
    return sim.acceleration(sim.q, sim.v, sim.time).cwiseAbs().maxCoeff();
}

} // namespace

// ============================================================================
// The motion ratio: geometry and virtual work
// ============================================================================

TEST_CASE("Suspensions: the motion ratio from the geometry equals the one from virtual work",
          "[vehicle][suspensions]")
{
    // The plan's test of task 7.6. For each layout, the corner alone on a
    // fixed chassis, a bump driver holding its wheel centre's height (a
    // solid axle's two, in heave). From the geometry: Kinematics' velocities
    // per unit of travel give the spring unit's rate of extension, the
    // motion ratio MR. From virtual work: without gravity, a spring unit
    // pushing apart with F0 = 1000 N alone, the corner at rest needs from
    // its drivers' multipliers lambda (M a + b = tau + J^T lambda; at rest
    // and along the motion's tangent v, J v = 1 for each driver) the power
    // the spring gives, sum(lambda) = -F0 sum(MR). Both come from the same
    // Jacobians by different paths (a velocity solve, a force solve): they
    // agree to rounding times the mechanism's conditioning; allow 1e-10 of
    // the larger.
    for (const Layout& l : layouts()) {
        kernel::System sys;
        const CornerRig rig = build_corner_rig(sys, l.tmpl, l.corner);
        Kinematics k(sys);
        REQUIRE(k.solve(50, 1e-12));
        const VecX v = k.velocities();
        kernel::Data data(sys.model);
        kernel::forward_kinematics(sys.model, data, k.q, v);
        Real geometry = 0.0;
        for (const CornerHandle& c : rig.corners) geometry += extension_rate(data, c.spring_end1, c.spring_end2);

        sys.model.gravity = Vec3::Zero();
        SpringDamperParams push;
        push.preload = 1000.0;
        for (const CornerHandle& c : rig.corners) {
            sys.force_elements.push_back(std::make_shared<SpringDamper>(
                c.spring_end1.body, c.spring_end2.body, c.spring_end1.X_BM.p, c.spring_end2.X_BM.p, push));
        }
        kernel::Simulator sim(sys);
        sim.q = k.q;
        sim.v.setZero();
        sim.refresh();
        const kernel::Loads loads = kernel::compute_loads(sim);
        // The multipliers are stacked in the constraints' order, each of its size.
        Real lambda = 0.0;
        for (std::size_t d : rig.drivers) {
            Index row = 0;
            for (std::size_t i = 0; i < d; ++i) row += sys.constraints[i]->size();
            lambda += loads.lambda(row);
        }
        const Real virtual_work = -lambda / 1000.0;
        INFO(l.name << ": motion ratio " << geometry << " from the geometry, " << virtual_work
             << " from virtual work; difference " << geometry - virtual_work);
        CHECK(std::abs(geometry) > 0.1);
        CHECK(std::abs(geometry - virtual_work) <= 1e-10 * std::max(std::abs(geometry), std::abs(virtual_work)));
    }
}

TEST_CASE("Suspensions: a trailing arm's motion ratio is its lever arms' ratio", "[vehicle][suspensions]")
{
    // The sedan's rear right trailing arm: its wheel centre W at (-1.35,
    // -0.8, 0), the hinge about Y through O = W + (0.45, 0.35, 0.02) (the
    // left offsets mirrored), the coil-over from S = W + (-0.05, 0.15, 0.32)
    // on the chassis to P = W + (-0.05, 0.15, 0) on the arm. Turning at w
    // about Y, a point moves at w (Y x r) = w (r_z, 0, -r_x), r from O:
    //   W - O = (-0.45, -0.35, -0.02): the wheel centre (-0.02, 0, 0.45) w,
    //   P - O = (-0.50, -0.20, -0.02): the arm's end (-0.02, 0, 0.50) w,
    // and the unit, along P - S = (0, 0, -0.32), shortens at 0.50 w. Per
    // unit of wheel rise: MR = -0.50 / 0.45 = -10/9, and the wheel centre
    // moves back by 0.02 / 0.45 = 0.0444. To rounding.
    const VehicleTemplate t = car(SuspensionType::Simple, SuspensionType::TrailingArm);
    kernel::System sys;
    const CornerRig rig = build_corner_rig(sys, t, 3);
    Kinematics k(sys);
    REQUIRE(k.solve(50, 1e-12));
    const VecX v = k.velocities();
    kernel::Data data(sys.model);
    kernel::forward_kinematics(sys.model, data, k.q, v);
    const CornerHandle& c = rig.corners[0];
    CHECK_THAT(extension_rate(data, c.spring_end1, c.spring_end2), WithinRel(-10.0 / 9.0, 1e-13));
    const RigidBodyState arm = kernel::body_state(data, c.upright_body);
    CHECK_THAT(arm.v_WB.z(), WithinRel(1.0, 1e-13));
    CHECK_THAT(arm.v_WB.x(), WithinRel(-0.02 / 0.45, 1e-12));
    CHECK_THAT(arm.w_WB.y(), WithinRel(1.0 / 0.45, 1e-13));
}

TEST_CASE("Suspensions: a pushrod's rocker multiplies the lower arm's turn by its levers",
          "[vehicle][suspensions]")
{
    // The sedan's front left double wishbone with a pushrod. The lower arm
    // turns about X through its pivot O_l; the pushrod from A on it to B on
    // the rocker, which turns about X through O_r; the unit from S on the
    // chassis to C on the rocker. All in the plane of the axle, so per unit
    // of the arm's turn (w = 1) A moves at X x (A - O_l); the pushrod's
    // length held, (v_B - v_A).(B - A) = 0, gives the rocker's turn
    //   w_r = [(X x (A - O_l)).(B - A)] / [(X x (B - O_r)).(B - A)],
    // and the unit extends at w_r (X x (C - O_r)).(C - S) / |C - S|. Times
    // the arm's turn per unit of wheel travel (from the velocities), that
    // is the motion ratio from the geometry: to rounding.
    VehicleTemplate t = car(SuspensionType::DoubleWishbone, SuspensionType::Simple);
    t.front_axle.spring_layout = SpringLayout::Pushrod;
    kernel::System sys;
    const CornerRig rig = build_corner_rig(sys, t, 0);
    Kinematics k(sys);
    REQUIRE(k.solve(50, 1e-12));
    const VecX v = k.velocities();
    kernel::Data data(sys.model);
    kernel::forward_kinematics(sys.model, data, k.q, v);
    const CornerHandle& c = rig.corners[0];
    REQUIRE(c.rocker_body > 0);

    const DwbHardpoints& h = t.front_axle.dwb;
    const Vec3 wc(t.front_axle_x, t.front_axle.half_track, 0.0);
    const Vec3 O_l = wc + h.lca_pivot_offset, A = wc + h.pushrod_arm_offset;
    const Vec3 O_r = wc + h.rocker_pivot_offset, B = wc + h.rocker_pushrod_offset;
    const Vec3 C = wc + h.rocker_spring_offset, S = wc + h.pushrod_spring_chassis_offset;
    const Vec3 X = Vec3::UnitX();
    const Real w_r = X.cross(A - O_l).dot(B - A) / X.cross(B - O_r).dot(B - A);
    const Real per_arm_turn = w_r * X.cross(C - O_r).dot((C - S).normalized());
    const Real arm_turn = kernel::body_state(data, c.lca_body).w_WB.x();
    const Real geometry = extension_rate(data, c.spring_end1, c.spring_end2);
    INFO("arm's turn " << arm_turn << " rad/m, rocker per arm turn " << w_r << ", unit per arm turn "
         << per_arm_turn << " m/rad; motion ratio " << geometry);
    CHECK_THAT(geometry, WithinRel(arm_turn * per_arm_turn, 1e-12));
    CHECK_THAT(kernel::body_state(data, c.rocker_body).w_WB.x(), WithinRel(arm_turn * w_r, 1e-12));
}

TEST_CASE("Suspensions: a solid axle rolls about a point on its Panhard rod's line", "[vehicle][suspensions]")
{
    // The sedan's rear solid axle on a fixed chassis, its left wheel centre
    // driven up and its right down (roll). The Panhard rod's axle end A
    // moves across the rod (its length held), so the axle's instantaneous
    // centre in the axle's plane lies on the rod's line: from the axle's
    // velocity v_o at its origin o and its roll rate w about X, the centre is
    // o + (-v_oz / w, v_oy / w) in (Y, Z), and its distance from the line
    // through the rod's ends is zero to rounding.
    const VehicleTemplate t = car(SuspensionType::Simple, SuspensionType::SolidAxle);
    kernel::System sys;
    sys.model.gravity = iso_gravity();
    const BodyIndex chassis = sys.model.add_body(0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
                                                 Transform3::Identity(),
                                                 RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.5, 0.2)), "chassis");
    const Real x = -t.rear_axle_x;
    const auto both = detail::build_solid_axle_corners(sys, chassis, x, t.rear_axle, "axle");
    for (int s = 0; s < 2; ++s) {
        const Real sign = s == 0 ? 1.0 : -1.0;
        sys.constraints.push_back(std::make_shared<kernel::Dot2>(
            kernel::Marker{0, Transform3::Identity()}, 2,
            kernel::Marker{both[static_cast<std::size_t>(s)].upright_body,
                           Transform3::FromTranslation(both[static_cast<std::size_t>(s)].wheel_center_in_upright)},
            kernel::TimeFunction([sign](Real tt) { return sign * tt; }, [sign](Real) { return sign; },
                                 [](Real) { return 0.0; })));
    }
    Kinematics k(sys);
    REQUIRE(k.solve(50, 1e-12));
    const VecX v = k.velocities();
    kernel::Data data(sys.model);
    kernel::forward_kinematics(sys.model, data, k.q, v);
    const RigidBodyState a = kernel::body_state(data, both[0].upright_body);
    const Real w = a.w_WB.x();
    const Vec3 centre(a.p_WB.x(), a.p_WB.y() - a.v_WB.z() / w, a.p_WB.z() + a.v_WB.y() / w);
    const SolidAxleParams p = detail::make_solid_axle_params(x, t.rear_axle);
    const Vec3 rod = (p.panhard_axle - p.panhard_chassis).normalized();
    const Vec3 off = centre - p.panhard_chassis;
    const Real distance = std::abs(off.y() * rod.z() - off.z() * rod.y());
    INFO("roll rate " << w << " rad/m; centre at y " << centre.y() << ", z " << centre.z() << " m; its distance from "
         << "the rod's line " << distance << " m");
    CHECK_THAT(w, WithinRel(2.0 / 1.6, 1e-12));   // the wheel centres 1.6 m apart, one up and one down
    CHECK(distance < 1e-14);
}

// ============================================================================
// Masses
// ============================================================================

TEST_CASE("Suspensions: arms and uprights carry their masses where their joints are", "[vehicle][suspensions]")
{
    // The sports car (double wishbones). Its front left lower arm, a rod of
    // 5 kg from its pivot to its ball joint (offsets (0, -0.40, -0.05) and
    // (0, -0.08, -0.10) from the wheel centre): its centre of mass midway,
    // (0, 0.16, -0.025) from the pivot in its frame, and 5 (0.015^2 * 3 +
    // L^2) / 12 across it, L = |(0, 0.32, -0.05)| = 0.32388. Its upright, 15
    // kg among its two ball joints, the tie rod's end and the wheel centre:
    // its centre of mass their centroid, (0.025, -0.065, -0.0125) from the
    // wheel centre. To rounding.
    const VehicleTemplate t = VehicleTemplate::SportsCar();
    kernel::System sys;
    const VehicleHandle vh = build_vehicle(sys, t);
    const CornerHandle& fl = vh.corners[0];
    const RigidBodyInertia& lca = sys.model.inertia[static_cast<std::size_t>(fl.lca_body)];
    CHECK(lca.mass == 5.0);
    CHECK((lca.com_B - Vec3(0.0, 0.16, -0.025)).norm() < 1e-15);
    const Vec3 u = Vec3(0.0, 0.32, -0.05).normalized();
    const Real across = 5.0 * (3.0 * 0.015 * 0.015 + 0.32 * 0.32 + 0.05 * 0.05) / 12.0;
    CHECK_THAT((lca.I_com_B * Vec3(1.0, 0.0, 0.0)).x(), WithinRel(across, 1e-14));
    CHECK_THAT(u.dot(lca.I_com_B * u), WithinRel(0.5 * 5.0 * 0.015 * 0.015, 1e-12));
    const RigidBodyInertia& upright = sys.model.inertia[static_cast<std::size_t>(fl.upright_body)];
    CHECK(upright.mass == 15.0);
    const Vec3 centroid = (Vec3(0.0, -0.08, -0.10) + Vec3(0.0, -0.10, 0.10) + Vec3(0.10, -0.08, -0.05)) / 4.0;
    CHECK((upright.com_B - centroid).norm() < 1e-15);

    // Every body counted: the template's total mass is the model's, for each
    // architecture and with the pushrod's rockers and a bar's halves.
    std::vector<VehicleTemplate> cars{VehicleTemplate::DefaultSedan(), VehicleTemplate::SportsCar(),
                                      VehicleTemplate::FWDHatchback()};
    for (const Layout& l : layouts()) cars.push_back(l.tmpl);
    VehicleTemplate bars = car(SuspensionType::DoubleWishbone, SuspensionType::SolidAxle);
    bars.front_axle.spring_layout = SpringLayout::Pushrod;
    bars.front_axle.arb_links.enabled = true;
    bars.rear_axle.arb_links.enabled = true;
    bars.front_axle.k_arb = 30000.0;
    bars.rear_axle.k_arb = 15000.0;
    cars.push_back(bars);
    for (const VehicleTemplate& c : cars) {
        kernel::System s;
        (void)build_vehicle(s, c);
        Real model_mass = 0.0;
        for (int b = 1; b < s.model.nbodies(); ++b) model_mass += s.model.inertia[static_cast<std::size_t>(b)].mass;
        CHECK_THAT(c.total_mass(), WithinRel(model_mass, 1e-15));
    }
}

// ============================================================================
// Equilibrium
// ============================================================================

TEST_CASE("Suspensions: each linkage car is placed at rest, its springs holding the design position",
          "[vehicle][suspensions]")
{
    // set_vehicle_equilibrium puts each linkage corner at its design
    // position, its spring set by virtual work to hold it there under the
    // tyre load the masses give and the tyre's overturning moment at rest;
    // a mixed car's simple corners at their springs' lengths under their
    // loads. What is left: the chassis is pitched so that both axles' wheel
    // centres sit at their loaded radii, theta = (R_r - R_f) / L, about
    // (200 N / 2.1e5 N/m) / 2.7 m = 3.5e-4 rad, and gravity, turned by it
    // into the chassis's axes, pushes the parts along X by g theta = 3.4e-3
    // m/s^2 of their weight, which the corners' motions take in part (they
    // move along X as they travel). On the lightest parts, a pushrod's
    // rocker (1.5 kg), that is up to ten times larger as an acceleration of
    // its coordinate: allow 0.05 m/s^2 (before task 7.6 the double-wishbone
    // sedan began at 73, F9). Settling it, statics moves the chassis by that
    // force over the wheel rate, about 1 N / 2.5e4 N/m: allow 1e-4 m, and it
    // converges in two Newton iterations without relaxation.
    std::vector<Layout> cars = layouts();
    cars.push_back({"sports car", VehicleTemplate::SportsCar(), 0});
    cars.push_back({"hatchback", VehicleTemplate::FWDHatchback(), 0});
    for (const Layout& l : cars) {
        kernel::System sys;
        const VehicleHandle vh = build_vehicle(sys, l.tmpl);
        kernel::Simulator sim(sys);
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        const Real placed = largest_acceleration(sim);
        const VecX q_placed = sim.q;
        const kernel::StaticReport r = kernel::static_equilibrium(sim);
        const int cq = sys.model.idx_q[static_cast<std::size_t>(vh.chassis_body)];
        const Real moved = (sim.q.segment<3>(cq) - q_placed.segment<3>(cq)).norm();
        INFO(l.name << ": largest acceleration as placed " << placed << ", statics " << r.iterations
             << " iterations, the chassis moved " << moved << " m; " << r.summary(sys.model));
        CHECK(r.converged);
        CHECK(r.relaxation_steps == 0);
        CHECK(r.iterations <= 2);
        CHECK(placed < 0.05);
        CHECK(moved < 1e-4);
    }
}


// ============================================================================
// The anti-roll bar on drop links
// ============================================================================

TEST_CASE("Suspensions: an anti-roll bar on drop links gives k_arb at the wheels in roll",
          "[vehicle][suspensions]")
{
    // The sports car's front: double wishbones, a bar on drop links asked
    // for k_arb = 30000 N/m, both corners on a fixed chassis with the bar's
    // halves and their torsion as build_vehicle makes them: k_t = k_arb /
    // r^2, r the half's turn per unit of its wheel's travel. The left wheel
    // centre is driven up by t, the right down by t. With theta = r z to
    // first order the bar's energy is (1/2) k_t (2 r t)^2 = 2 k_arb t^2, as
    // the force element's (1/2) k_arb (z_L - z_R)^2: its second derivative,
    // the bar's stiffness in this roll, is 4 k_arb. Without gravity the
    // drivers' multipliers carry the bar's force: lambda_L - lambda_R = dV/dt
    // (the drivers' rates being +1 and -1). V is even in t (the car's mirror
    // symmetry), so (dV/dt)(delta) / delta = V''(0) + V''''(0) delta^2 / 6:
    // with delta = 1e-5 m and V varying over the arms' lengths (0.2 m), the
    // second term is about (1e-5 / 0.2)^2 = 2.5e-9 relative; allow 1e-7.
    VehicleTemplate t = VehicleTemplate::SportsCar();
    AxleConfig& ax = t.front_axle;
    ax.arb_links.enabled = true;
    ax.k_arb = 30000.0;
    ax.is_steered = false;
    kernel::System sys;
    sys.model.gravity = Vec3::Zero();
    const BodyIndex chassis = sys.model.add_body(0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
                                                 Transform3::Identity(),
                                                 RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.5, 0.2)), "chassis");
    const Vec3 left_wc(t.front_axle_x, ax.half_track, 0.0);
    std::array<CornerHandle, 2> c{detail::build_linkage_corner(sys, chassis, left_wc, ax, false, chassis, "L"),
                                  detail::build_linkage_corner(sys, chassis, detail::mirror_y(left_wc), ax, true,
                                                               chassis, "R")};
    detail::build_anti_roll_bar_half(sys, chassis, left_wc, ax, c[0], false);
    detail::build_anti_roll_bar_half(sys, chassis, left_wc, ax, c[1], true);
    const TyreBody tyre = ax.tyre_body();
    const detail::CornerMotion m = detail::corner_motion(t, 0, tyre.free_radius);
    const Real r = m.bar_rate[0];
    RotationalSpringDamperParams tp;
    tp.spring = Curve::linear(ax.k_arb / (r * r));
    const Transform3 along_y = Transform3::FromRotation(detail::rotation_align_z_to(Vec3::UnitY()));
    sys.force_elements.push_back(
        std::make_shared<RotationalSpringDamper>(c[0].arb_half_body, along_y, c[1].arb_half_body, along_y, tp));
    std::array<Index, 2> rows{};
    for (int s = 0; s < 2; ++s) {
        Index row = 0;
        for (const auto& constraint : sys.constraints) row += constraint->size();
        rows[static_cast<std::size_t>(s)] = row;
        const Real sign = s == 0 ? 1.0 : -1.0;
        sys.constraints.push_back(std::make_shared<kernel::Dot2>(
            kernel::Marker{0, Transform3::Identity()}, 2,
            kernel::Marker{c[static_cast<std::size_t>(s)].upright_body, Transform3::Identity()},
            kernel::TimeFunction([sign](Real tt) { return sign * tt; }, [sign](Real) { return sign; },
                                 [](Real) { return 0.0; })));
    }
    const Real delta = 1e-5;
    Kinematics k(sys);
    k.t = delta;
    REQUIRE(k.solve(50, 1e-13));
    kernel::Simulator sim(sys);
    sim.time = delta;
    sim.q = k.q;
    sim.v.setZero();
    sim.refresh();
    const kernel::Loads loads = kernel::compute_loads(sim);
    const Real stiffness = (loads.lambda(rows[0]) - loads.lambda(rows[1])) / delta;
    INFO("the half's turn " << r << " rad/m, the bar " << ax.k_arb / (r * r) << " N m/rad; roll stiffness "
         << stiffness << " N/m against 4 k_arb = " << 4.0 * ax.k_arb << "; relative difference "
         << stiffness / (4.0 * ax.k_arb) - 1.0);
    CHECK_THAT(stiffness, WithinRel(4.0 * ax.k_arb, 1e-7));
}

// ============================================================================
// The spring unit
// ============================================================================

TEST_CASE("Suspensions: a spring unit's damper curve and stops, measured from its design length",
          "[vehicle][suspensions]")
{
    // The sports car's front with a damper table, a bump stop after 30 mm
    // (1e5 N/m) and a rebound stop after 50 mm (2e5 N/m). At the design
    // length the spring pushes with the design force; 40 mm shorter the bump
    // stop pushes with 1e5 * 0.01 = 1000 N, 60 mm longer the rebound stop
    // pulls with 2e5 * 0.01 = 2000 N, and between them the stops carry
    // nothing. The damper's force at 0.5 m/s of extension is the table's,
    // opposing it. To rounding.
    VehicleTemplate t = VehicleTemplate::SportsCar();
    AxleConfig& ax = t.front_axle;
    ax.damper_curve = Curve::table({-1.0, -0.2, 0.0, 0.2, 1.0}, {-1500.0, -500.0, 0.0, 800.0, 2000.0});
    ax.bump_travel = 0.03;
    ax.bump_stop = Curve::linear(1e5);
    ax.rebound_travel = 0.05;
    ax.rebound_stop = Curve::linear(2e5);
    kernel::System sys;
    const VehicleHandle vh = build_vehicle(sys, t);
    const CornerHandle& fl = vh.corners[0];
    REQUIRE(fl.spring != nullptr);
    REQUIRE(fl.stops != nullptr);
    CHECK(fl.top_mount == nullptr);
    const Real L = fl.design_spring_length;
    CHECK_THAT(fl.spring->law(L, 0.0).force, WithinRel(fl.design_spring_force, 1e-14));
    CHECK_THAT(fl.spring->law(L, 0.5).force, WithinRel(fl.design_spring_force - ax.damper_curve.value(0.5), 1e-14));
    CHECK(fl.stops->params().free_length == L);
    CHECK_THAT(fl.stops->law(L - 0.04, 0.0).force, WithinRel(1000.0, 1e-9));
    CHECK_THAT(fl.stops->law(L + 0.06, 0.0).force, WithinRel(-2000.0, 1e-9));
    CHECK(fl.stops->law(L - 0.02, 0.0).force == 0.0);
    CHECK(fl.stops->law(L + 0.04, 0.0).force == 0.0);
    // The rear axle has none of them: no stops element.
    CHECK(vh.corners[2].stops == nullptr);
}

TEST_CASE("Suspensions: a damper on a top mount leaves the car resting where it rests without one",
          "[vehicle][suspensions]")
{
    // The sports car with each damper in series with a top mount of 2e6
    // N/m (a MaxwellDamper): the unit's spring alone in the SpringDamper.
    // The Maxwell element carries no force at rest (D66), so statics finds
    // the same equilibrium as without top mounts, to statics' own error
    // (accelerations below 1e-6, over the softest supported mode's 1 Hz:
    // 2.5e-8 m); allow 1e-7 m on every coordinate.
    VehicleTemplate t = VehicleTemplate::SportsCar();
    std::array<VecX, 2> q;
    for (int with = 0; with < 2; ++with) {
        if (with == 1) {
            t.front_axle.top_mount_stiffness = 2e6;
            t.rear_axle.top_mount_stiffness = 2e6;
        }
        kernel::System sys;
        const VehicleHandle vh = build_vehicle(sys, t);
        if (with == 1) {
            REQUIRE(vh.corners[0].top_mount != nullptr);
            CHECK(vh.corners[0].spring->params().damper.is_zero());
        }
        kernel::Simulator sim(sys);
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        REQUIRE(kernel::static_equilibrium(sim).converged);
        q[static_cast<std::size_t>(with)] = sim.q;
    }
    INFO("largest difference " << (q[0] - q[1]).cwiseAbs().maxCoeff());
    CHECK((q[0] - q[1]).cwiseAbs().maxCoeff() < 1e-7);
}

// ============================================================================
// The new architectures in a car
// ============================================================================

TEST_CASE("Suspensions: cars on every architecture have fourteen freedoms and stay at rest",
          "[vehicle][suspensions]")
{
    // A pushrod front with a five-link rear; McPherson with a trailing-arm
    // rear and bars on drop links on both axles; double wishbones with a
    // solid axle and bars. Each keeps the chassis's six freedoms and one
    // travel and one spin per wheel (a solid axle's two: heave and roll),
    // all else removed by independent equations. Placed and settled by
    // statics, a second of simulation (RK4 at 0.5 ms) moves the chassis by
    // under a micrometre: with accelerations below 1e-6, a mode of frequency
    // w moves by at most 2e-6 / w^2, 5e-8 m at the body's 1 Hz.
    std::vector<VehicleTemplate> cars;
    VehicleTemplate a = car(SuspensionType::DoubleWishbone, SuspensionType::FiveLink);
    a.front_axle.spring_layout = SpringLayout::Pushrod;
    cars.push_back(a);
    VehicleTemplate b = car(SuspensionType::McPherson, SuspensionType::TrailingArm);
    b.front_axle.arb_links.enabled = true;
    b.front_axle.k_arb = 25000.0;
    b.rear_axle.arb_links.enabled = true;
    b.rear_axle.k_arb = 10000.0;
    cars.push_back(b);
    VehicleTemplate c = car(SuspensionType::DoubleWishbone, SuspensionType::SolidAxle);
    c.front_axle.arb_links.enabled = true;
    c.front_axle.k_arb = 30000.0;
    c.rear_axle.arb_links.enabled = true;
    c.rear_axle.k_arb = 15000.0;
    cars.push_back(c);
    for (const VehicleTemplate& tmpl : cars) {
        kernel::System sys;
        const VehicleHandle vh = build_vehicle(sys, tmpl);
        const kernel::ValidationReport r = kernel::validate(sys);
        INFO(r.summary());
        CHECK(r.ok());
        CHECK(r.warnings.empty());
        CHECK(r.redundant_equations == 0);
        CHECK(r.degrees_of_freedom == 14);
        kernel::Simulator sim(sys);
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        REQUIRE(kernel::static_equilibrium(sim).converged);
        const int cq = sys.model.idx_q[static_cast<std::size_t>(vh.chassis_body)];
        const Vec3 start = sim.q.segment<3>(cq);
        sim.run(1.0, 5e-4);
        const Real drift = (sim.q.segment<3>(cq) - start).norm();
        INFO("chassis drift over 1 s: " << drift << " m");
        CHECK(drift < 1e-6);
    }
}

TEST_CASE("Suspensions: what the builder cannot make is refused", "[vehicle][suspensions]")
{
    auto error_of = [](const VehicleTemplate& t) {
        try {
            kernel::System sys;
            (void)build_vehicle(sys, t);
        } catch (const MbdError& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    VehicleTemplate steered_arm = car(SuspensionType::TrailingArm, SuspensionType::Simple);
    CHECK_THAT(error_of(steered_arm), ContainsSubstring("MBD-V002"));
    VehicleTemplate steered_axle = car(SuspensionType::SolidAxle, SuspensionType::Simple);
    CHECK_THAT(error_of(steered_axle), ContainsSubstring("MBD-V002"));
    VehicleTemplate pushrod_strut = car(SuspensionType::McPherson, SuspensionType::Simple);
    pushrod_strut.front_axle.spring_layout = SpringLayout::Pushrod;
    CHECK_THAT(error_of(pushrod_strut), ContainsSubstring("MBD-V002"));
    VehicleTemplate no_spring = car(SuspensionType::DoubleWishbone, SuspensionType::Simple);
    no_spring.front_axle.k_spring = 0.0;
    CHECK_THAT(error_of(no_spring), ContainsSubstring("MBD-V004"));
}
