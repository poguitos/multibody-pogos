// Task 7.7 (D68): the steering. A rack geared to a pinion, a compliant column
// to a steering wheel, the input as the wheel's angle or a torque on it, and
// the tie rods as links from the rack to the uprights. The plan's test:
// Ackermann behaviour emerges from the geometry (the linkage's own angles,
// against the closed form of the planar linkage, no Ackermann formula in the
// code), and the rack moves as one piece. Expected values are closed forms
// written out here.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/validate.hpp"
#include "mbd/vehicle/axes.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace {

/// A chassis on a fixed joint at the origin, without gravity.
BodyIndex fixed_chassis(kernel::System& sys)
{
    sys.model.gravity = Vec3::Zero();
    return sys.model.add_body(0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
                              Transform3::Identity(), RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.5, 0.2)),
                              "chassis");
}

/// A constraint holding a body point's height (world Z) at zero.
std::shared_ptr<kernel::ConstraintModel> hold_height(BodyIndex body)
{
    return std::make_shared<kernel::Dot2>(kernel::Marker{0, Transform3::Identity()}, 2,
                                          kernel::Marker{body, Transform3::Identity()},
                                          kernel::TimeFunction::constant(0.0));
}

/// The steering system alone: a rack, its column and a spring of k_r on the
/// rack's travel, on a fixed chassis.
struct ColumnRig {
    kernel::System sys;
    BodyIndex rack{0};
    SteeringColumn column;
};

void build_column_rig(ColumnRig& rig, SteeringInput input, Real k_r, Real damping)
{
    const BodyIndex chassis = fixed_chassis(rig.sys);
    rig.rack = detail::add_steering_rack(rig.sys, chassis, nullptr, "rack");
    SteeringConfig config;
    config.input = input;
    config.column_damping = damping;
    rig.column = detail::add_steering_column(rig.sys, chassis, rig.rack, config, 0.008, 0.05);
    kernel::JointCoordinateForceParams spring;
    spring.spring = Curve::linear(k_r);
    rig.sys.joint_forces.push_back(std::make_shared<kernel::JointCoordinateForce>(rig.sys.model, rig.rack, spring));
}

Real coordinate(const kernel::Simulator& sim, BodyIndex body)
{
    return sim.q(sim.system.model.idx_q[static_cast<std::size_t>(body)]);
}

Real rate(const kernel::Simulator& sim, BodyIndex body)
{
    return sim.v(sim.system.model.idx_v[static_cast<std::size_t>(body)]);
}

/// The angle delta by which a wheel turns about its vertical kingpin K for
/// the rack's travel s, in the planar linkage: its steering arm d = P0 - K
/// turns with it, P = K + Rz(delta) d, and the tie rod from the rack's end
/// I = I0 + (0, s) keeps its length l = |P0 - I0|:
///   |Rz(delta) d + e|^2 = l^2,  e = K - I,
/// so A cos(delta) + B sin(delta) = C with A = e.d, B = e_y d_x - e_x d_y,
/// C = (l^2 - |d|^2 - |e|^2) / 2: delta = atan2(B, A) -+ acos(C / hypot(A, B)),
/// the root nearest `previous`.
Real linkage_angle(const Vec3& K, const Vec3& d, const Vec3& I0, Real s, Real previous)
{
    const Real l2 = (K + d - I0).head<2>().squaredNorm();
    const Vec3 e = K - (I0 + Vec3(0.0, s, 0.0));
    const Real A = e.x() * d.x() + e.y() * d.y();
    const Real B = e.y() * d.x() - e.x() * d.y();
    const Real C = 0.5 * (l2 - d.head<2>().squaredNorm() - e.head<2>().squaredNorm());
    const Real phi = std::atan2(B, A);
    const Real c = std::acos(std::clamp(C / std::hypot(A, B), Real(-1.0), Real(1.0)));
    const Real r1 = std::remainder(phi - c, 2.0 * pi), r2 = std::remainder(phi + c, 2.0 * pi);
    return std::abs(r1 - previous) < std::abs(r2 - previous) ? r1 : r2;
}

} // namespace

// ============================================================================
// The column, the pinion and the rack
// ============================================================================

TEST_CASE("Steering: a held steering wheel and a held torque against the column's and the rack's statics",
          "[vehicle][steering][column]")
{
    // The pinion's ratio r = 0.008 m/rad, the column k_c = 2000 N m/rad, a
    // spring k_r = 2e4 N/m on the rack's travel s = r theta_p.
    //   The wheel held at theta = 0.5 rad: the column's torque k_c (theta -
    //   theta_p) is the rack's force times r, k_r r^2 theta_p, so theta_p =
    //   k_c theta / (k_c + k_r r^2) = 1000 / 2001.28 = 0.499680 rad.
    //   A torque T = 2 N m on the wheel: the rack's force T / r, s = T / (r
    //   k_r) = 0.0125 m, theta_p = s / r = 1.5625 rad, the wheel T / k_c =
    //   0.001 rad further.
    // Statics' tolerance (accelerations below 1e-6, over stiffnesses of
    // 1.28 N m/rad on 2.6e-4 kg m^2 at the least, 4800 s^-2) leaves 2e-10
    // rad: allow 1e-9 relative.
    const Real r = 0.008, k_c = 2000.0, k_r = 2e4;
    {
        ColumnRig rig;
        build_column_rig(rig, SteeringInput::WheelAngle, k_r, 2.0);
        kernel::Simulator sim(rig.sys);
        rig.column.command->hold(0.5);
        sim.initialize();
        const kernel::StaticReport report = kernel::static_equilibrium(sim);
        INFO(report.summary(rig.sys.model));
        REQUIRE(report.converged);
        const Real theta_p = k_c * 0.5 / (k_c + k_r * r * r);
        CHECK_THAT(coordinate(sim, rig.column.wheel_body), WithinRel(0.5, 1e-12));
        CHECK_THAT(coordinate(sim, rig.column.pinion_body), WithinRel(theta_p, 1e-9));
        CHECK_THAT(coordinate(sim, rig.rack), WithinRel(r * theta_p, 1e-9));
    }
    {
        ColumnRig rig;
        build_column_rig(rig, SteeringInput::WheelTorque, k_r, 2.0);
        kernel::Simulator sim(rig.sys);
        rig.column.command->hold(2.0);
        sim.initialize();
        const kernel::StaticReport report = kernel::static_equilibrium(sim);
        INFO(report.summary(rig.sys.model));
        REQUIRE(report.converged);
        CHECK_THAT(coordinate(sim, rig.rack), WithinRel(2.0 / (r * k_r), 1e-9));
        CHECK_THAT(coordinate(sim, rig.column.pinion_body), WithinRel(1.5625, 1e-9));
        CHECK_THAT(coordinate(sim, rig.column.wheel_body), WithinRel(1.5625 + 2.0 / k_c, 1e-9));
    }
}

TEST_CASE("Steering: the column's two modes are those of its two inertias on its two springs",
          "[vehicle][steering][column]")
{
    // A torque-steered column, its damping off: the steering wheel I_w = 0.04
    // kg m^2, the pinion with the rack J = I_p + m_r r^2 = 2e-4 + 1 * 0.008^2 =
    // 2.64e-4 kg m^2 (in the pinion's angle), the column k_c = 2000 N m/rad
    // between them, the rack's spring k_r r^2 = 1.28 N m/rad on the pinion.
    // det(K - w^2 M) = 0: I_w J w^4 - (k_c J + (k_c + k_r r^2) I_w) w^2 + k_c
    // k_r r^2 = 0, two frequencies. The linearisation's differences hold them
    // to about 1e-7 relative (D27); allow 1e-6.
    ColumnRig rig;
    build_column_rig(rig, SteeringInput::WheelTorque, 2e4, 0.0);
    kernel::Simulator sim(rig.sys);
    sim.initialize();
    const kernel::LinearModel lin = kernel::linearize(sim);
    const Real I_w = 0.04, J = 2e-4 + 1.0 * 0.008 * 0.008, k_c = 2000.0, kr2 = 2e4 * 0.008 * 0.008;
    const Real a = I_w * J, b = k_c * J + (k_c + kr2) * I_w, c = k_c * kr2;
    const Real disc = std::sqrt(b * b - 4.0 * a * c);
    const std::array<Real, 2> w{std::sqrt((b - disc) / (2.0 * a)), std::sqrt((b + disc) / (2.0 * a))};
    std::vector<Real> found;
    for (const kernel::Mode& mode : lin.modes) {
        if (mode.oscillatory()) found.push_back(mode.natural_frequency);   // one of each conjugate pair
    }
    std::sort(found.begin(), found.end());
    INFO("closed form " << w[0] << " and " << w[1] << " rad/s; " << lin.summary(rig.sys.model));
    REQUIRE(found.size() == 2);
    CHECK_THAT(found[0], WithinRel(w[0], 1e-6));
    CHECK_THAT(found[1], WithinRel(w[1], 1e-6));
}

TEST_CASE("Steering: the rack moves as one piece, geared to the pinion, its rate in the equations",
          "[vehicle][steering][rack]")
{
    // The sports car's front axle on a fixed chassis, its wheel centres held
    // at their height, steered by its wheel from 0 to 0.5 rad over 1 s along
    // the quintic 10 t^3 - 15 t^4 + 6 t^5 (its rate and acceleration zero at
    // both ends, so that the column, whose mode with the rack and the
    // uprights is some 470 rad/s by estimate, is not jolted), RK4 at 0.5 ms,
    // no gravity.
    // At every step:
    //   - both tie rods' inner points move by the rack's travel: one piece;
    //   - the travel is the pinion's ratio times the pinion's angle, to the
    //     projection's 1e-10 m, and its rate the ratio times the pinion's
    //     rate, to rounding: the coupler at position and velocity level;
    //   - that rate is the travel's rate of change: against the recorded
    //     travel's central difference, which errs by h^2 s''' / 6, s''' at
    //     most r 0.5 60 = 0.2 m/s^3 (r = 0.0067 m/rad): 8e-9 m/s; allow 1e-7.
    //     (Before task 7.7 a rack set between steps moved with a rate of
    //     zero, F10.)
    const VehicleTemplate t = VehicleTemplate::SportsCar();
    const AxleConfig& ax = t.front_axle;
    kernel::System sys;
    const BodyIndex chassis = fixed_chassis(sys);
    const BodyIndex rack = detail::add_steering_rack(sys, chassis, nullptr, "rack");
    const Vec3 left(t.front_axle_x, ax.half_track, 0.0);
    std::array<CornerHandle, 2> c{detail::build_linkage_corner(sys, chassis, left, ax, false, rack, "L"),
                                  detail::build_linkage_corner(sys, chassis, detail::mirror_y(left), ax, true, rack,
                                                               "R")};
    for (const CornerHandle& corner : c) sys.constraints.push_back(hold_height(corner.upright_body));
    const Real per_rad = 0.5 * (detail::calibrate_rack(t, ax, left, false)
                                + detail::calibrate_rack(t, ax, detail::mirror_y(left), true));
    SteeringConfig config = t.steering;
    config.input = SteeringInput::WheelAngle;
    const SteeringColumn col = detail::add_steering_column(sys, chassis, rack, config,
                                                           per_rad / config.steering_ratio,
                                                           config.max_steer_angle * std::abs(per_rad));
    col.command->follow(kernel::TimeFunction(
        [](Real tt) { return 0.5 * tt * tt * tt * (10.0 - 15.0 * tt + 6.0 * tt * tt); },
        [](Real tt) { return 0.5 * 30.0 * tt * tt * (1.0 - tt) * (1.0 - tt); },
        [](Real tt) { return 0.5 * 60.0 * tt * (1.0 - tt) * (1.0 - 2.0 * tt); }));
    // The tie rods' inner points, on the rack.
    std::array<kernel::Marker, 2> inner;
    for (std::size_t s = 0; s < 2; ++s) {
        const auto* tie = dynamic_cast<const kernel::Distance*>(sys.constraints[c[s].tierod_constraint_idx].get());
        REQUIRE(tie != nullptr);
        REQUIRE(tie->i().body == rack);
        inner[s] = tie->i();
    }
    kernel::Simulator sim(sys);
    sim.initialize();
    auto inner_point = [&](std::size_t s) {
        const RigidBodyState& r = sim.states()[static_cast<std::size_t>(rack)];
        return Vec3(r.p_WB + r.q_WB * inner[s].X_BM.p);
    };
    const std::array<Vec3, 2> start{inner_point(0), inner_point(1)};
    const Real h = 5e-4;
    std::vector<Real> travel{coordinate(sim, rack)}, speed{rate(sim, rack)};
    Real worst_piece = 0.0, worst_gear = 0.0, worst_gear_rate = 0.0, largest_toe = 0.0;
    for (int i = 0; i < 2000; ++i) {
        sim.step(h);
        const Real s = coordinate(sim, rack);
        travel.push_back(s);
        speed.push_back(rate(sim, rack));
        for (std::size_t k = 0; k < 2; ++k) {
            worst_piece = std::max(worst_piece, (inner_point(k) - start[k] - Vec3(0.0, s, 0.0)).norm());
        }
        worst_gear = std::max(worst_gear, std::abs(s - col.pinion_ratio * coordinate(sim, col.pinion_body)));
        worst_gear_rate = std::max(worst_gear_rate,
                                   std::abs(rate(sim, rack) - col.pinion_ratio * rate(sim, col.pinion_body)));
        largest_toe = std::max(largest_toe, std::abs(extract_toe(sim.states()[static_cast<std::size_t>(c[0].upright_body)])));
    }
    Real worst_rate = 0.0, largest_speed = 0.0;
    for (std::size_t i = 1; i + 1 < travel.size(); ++i) {
        worst_rate = std::max(worst_rate, std::abs(speed[i] - (travel[i + 1] - travel[i - 1]) / (2.0 * h)));
        largest_speed = std::max(largest_speed, std::abs(speed[i]));
    }
    INFO("rack: one piece within " << worst_piece << " m; geared within " << worst_gear << " m and "
         << worst_gear_rate << " m/s; its rate against its travel's within " << worst_rate << " m/s (largest "
         << largest_speed << " m/s); the left wheel's toe up to " << largest_toe << " rad");
    CHECK(worst_piece < 1e-12);
    CHECK(worst_gear < 1e-10);
    CHECK(worst_gear_rate < 1e-12);
    CHECK(worst_rate < 1e-7);
    CHECK(largest_speed > 0.003);   // the ramp's peak, r 0.5 15/8 = 6.3 mm/s, less the column's lag
    // The road wheels steer: at the peak about the 0.5 / 15 rad the steering
    // ratio gives at the centre (a check of the direction and size only: the
    // linkage's curvature and the column's twist change it).
    CHECK(largest_toe > 0.5 / config.steering_ratio * 0.8);
}

// ============================================================================
// Ackermann from the geometry
// ============================================================================

TEST_CASE("Steering: Ackermann behaviour emerges from the steering arms' geometry", "[vehicle][steering][ackermann]")
{
    // The sedan's front with double wishbones whose kingpins are vertical
    // (both ball joints at (0, -0.09) from the wheel centre) and whose tie
    // rods are level, at -0.05: the uprights turn about vertical axes and the
    // rack, the tie rods and the steering arms form a planar linkage on each
    // side, whose angles are linkage_angle's closed form. Three arms of 0.12
    // m from the kingpin: straight ahead (parallel), turned outward along the
    // line from the rear axle's centre through the kingpin (Ackermann's
    // rule), and turned inward as much (anti-Ackermann); each tie rod's rack
    // end level with its arm's end, 0.42 m inboard of the wheel centre.
    //
    // Swept through rack travels of 0 to 60 mm (a left turn), the
    // mechanism's toes are the closed form's to 1e-10 rad (Kinematics to
    // 1e-12 over a lever of 0.12 m). The behaviour then follows from the
    // closed form alone, and the code has no Ackermann formula: with the arms
    // turned outward the inner (left) wheel steers more and the wheels' axes
    // meet behind the front axle, on the left; with parallel arms the two
    // angles differ only at second order; turned inward the outer wheel
    // steers more and the axes meet in front. Reported: each one's share of
    // the ideal Ackermann difference, cot(delta_o) - cot(delta_i) = t / L.
    const Real x_f = 1.35, x_r = -1.35, ht = 0.8;
    const Real ky = ht - 0.09;
    const Vec3 to_kingpin = Vec3(x_f - x_r, ky, 0.0).normalized();
    struct Case {
        std::string name;
        Vec3 arm;
    };
    const std::vector<Case> cases{{"parallel", Vec3(0.12, 0.0, 0.0)},
                                  {"outward (Ackermann)", 0.12 * to_kingpin},
                                  {"inward (anti-Ackermann)", 0.12 * Vec3(to_kingpin.x(), -to_kingpin.y(), 0.0)}};
    for (const Case& cs : cases) {
        VehicleTemplate t = VehicleTemplate::DefaultSedan();
        AxleConfig& ax = t.front_axle;
        ax.suspension_type = SuspensionType::DoubleWishbone;
        ax.dwb.lca_outer_offset = Vec3(0.0, -0.09, -0.10);
        ax.dwb.uca_outer_offset = Vec3(0.0, -0.09, 0.10);
        ax.dwb.tierod_outer_offset = Vec3(cs.arm.x(), -0.09 + cs.arm.y(), -0.05);
        ax.dwb.tierod_inner_offset = Vec3(cs.arm.x(), -0.42, -0.05);

        kernel::System sys;
        const BodyIndex chassis = fixed_chassis(sys);
        auto command = std::make_shared<SteeringRack>();
        command->follow(kernel::TimeFunction([](Real tt) { return tt; }, [](Real) { return 1.0; },
                                             [](Real) { return 0.0; }));
        const BodyIndex rack = detail::add_steering_rack(sys, chassis, command, "rack");
        const Vec3 left(x_f, ht, 0.0);
        std::array<CornerHandle, 2> c{detail::build_linkage_corner(sys, chassis, left, ax, false, rack, "L"),
                                      detail::build_linkage_corner(sys, chassis, detail::mirror_y(left), ax, true,
                                                                   rack, "R")};
        for (const CornerHandle& corner : c) sys.constraints.push_back(hold_height(corner.upright_body));
        Kinematics k(sys);
        // Each side's kingpin, arm and rack end in the plane.
        const Vec3 K_L(x_f, ky, 0.0), K_R(x_f, -ky, 0.0);
        const Vec3 d_L(cs.arm.x(), cs.arm.y(), 0.0), d_R(cs.arm.x(), -cs.arm.y(), 0.0);
        const Vec3 I_L(x_f + cs.arm.x(), ht - 0.42, 0.0), I_R(x_f + cs.arm.x(), -(ht - 0.42), 0.0);
        Real dl = 0.0, dr = 0.0, worst = 0.0;
        Real toe_L = 0.0, toe_R = 0.0;
        for (int i = 1; i <= 60; ++i) {
            k.t = 0.001 * i;
            REQUIRE(k.solve(50, 1e-12));
            toe_L = extract_toe(k.state(c[0].upright_body));
            toe_R = extract_toe(k.state(c[1].upright_body));
            dl = linkage_angle(K_L, d_L, I_L, k.t, dl);
            dr = linkage_angle(K_R, d_R, I_R, k.t, dr);
            worst = std::max({worst, std::abs(toe_L - dl), std::abs(toe_R - dr)});
        }
        // At 60 mm: where the wheels' axes meet, each wheel centre turned
        // about its kingpin with its upright.
        const Vec3 W_L = K_L + Vec3(-0.09 * std::sin(toe_L), 0.09 * std::cos(toe_L), 0.0);
        const Vec3 W_R = K_R + Vec3(0.09 * std::sin(toe_R), -0.09 * std::cos(toe_R), 0.0);
        const Vec3 n_L(-std::sin(toe_L), std::cos(toe_L), 0.0), n_R(-std::sin(toe_R), std::cos(toe_R), 0.0);
        // W_L + a n_L = W_R + b n_R, in the plane.
        const Real det = -n_L.x() * n_R.y() + n_R.x() * n_L.y();
        const Vec3 gap = W_R - W_L;
        const Real a = (-gap.x() * n_R.y() + n_R.x() * gap.y()) / det;
        const Vec3 centre = W_L + a * n_L;
        const Real ideal_outer = std::atan(1.0 / (1.0 / std::tan(toe_L) + 2.0 * ky / (x_f - x_r)));
        const Real share = (toe_L - toe_R) / (toe_L - ideal_outer);
        INFO(cs.name << ": toes against the closed form within " << worst << " rad; at 60 mm the left "
             << toe_L << " rad, the right " << toe_R << "; the axes meet at x " << centre.x() << ", y "
             << centre.y() << " m; " << 100.0 * share << " % of Ackermann");
        CHECK(worst < 1e-10);
        CHECK(toe_L > 0.4);   // a left turn, about 0.5 rad
        if (cs.name == "parallel") {
            // Equal to first order: their difference grows with the square
            // of the angle, 7 % of the ideal one at 0.5 rad by the closed form.
            CHECK(std::abs(share) < 0.1);
        } else if (cs.name == "outward (Ackermann)") {
            CHECK(toe_L > toe_R);
            CHECK(centre.x() < x_f);
            CHECK(centre.y() > 0.0);
            CHECK(share > 0.3);
        } else {
            CHECK(toe_R > toe_L);
            CHECK(centre.x() > x_f);
            CHECK(share < -0.3);
        }
    }
}

// ============================================================================
// The car
// ============================================================================

TEST_CASE("Steering: a car steered by its wheel's angle or by a torque", "[vehicle][steering][car]")
{
    // The sports car three ways. By the rack's travel: fourteen freedoms, as
    // before (the rack driven). By the steering wheel's angle: the rack, the
    // pinion and the wheel, less the coupler and the wheel's driver, leave
    // one, the column's twist: fifteen. By a torque: sixteen, the steering
    // and the twist. The pinion's ratio is the linkage's mean
    // rack travel per radian (calibrated over the first 5 mm) over the
    // steering ratio, so that the steering wheel turns steering_ratio times
    // the road wheels there; the rack's stops at max_steer_angle. Settled,
    // then steered left by the wheel's angle (a ramp) or by a torque of 1 N m,
    // the car's left wheel turns left.
    for (const SteeringInput input : {SteeringInput::RackTravel, SteeringInput::WheelAngle,
                                      SteeringInput::WheelTorque}) {
        VehicleTemplate t = VehicleTemplate::SportsCar();
        t.steering.input = input;
        kernel::System sys;
        VehicleHandle vh = build_vehicle(sys, t);
        const kernel::ValidationReport r = kernel::validate(sys);
        INFO(r.summary());
        CHECK(r.ok());
        CHECK(r.redundant_equations == 0);
        CHECK(r.degrees_of_freedom
              == (input == SteeringInput::RackTravel ? 14 : input == SteeringInput::WheelAngle ? 15 : 16));
        if (input == SteeringInput::RackTravel) {
            CHECK(vh.column.wheel_body == 0);
            continue;
        }
        REQUIRE(vh.column.wheel_body > 0);
        const Real per_rad = 0.5 * (vh.corners[0].rack_per_rad + vh.corners[1].rack_per_rad);
        CHECK_THAT(vh.column.pinion_ratio, WithinRel(per_rad / t.steering.steering_ratio, 1e-15));
        CHECK_THAT(vh.column.rack_limit, WithinRel(t.steering.max_steer_angle * std::abs(per_rad), 1e-15));

        // At rest and settled, then steered: the car turns left.
        kernel::Simulator sim(sys);
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        REQUIRE(kernel::static_equilibrium(sim).converged);
        if (input == SteeringInput::WheelAngle) {
            vh.steer(kernel::TimeFunction([](Real tt) { return 0.3 * std::min(tt, 1.0); },
                                          [](Real tt) { return tt < 1.0 ? 0.3 : 0.0; },
                                          [](Real) { return 0.0; }));
        } else {
            vh.set_steering_torque(1.0);
        }
        sim.run(0.5, 5e-4);
        const Real toe_L = extract_toe(sim.states()[static_cast<std::size_t>(vh.corners[0].upright_body)]);
        INFO("after 0.5 s the left wheel's toe " << toe_L << " rad");
        CHECK(toe_L > 0.0);
    }

    // A torque-steered car refuses an angle, and a rack-steered one a torque.
    VehicleTemplate t = VehicleTemplate::SportsCar();
    t.steering.input = SteeringInput::WheelTorque;
    kernel::System sys;
    VehicleHandle vh = build_vehicle(sys, t);
    auto error_of = [](auto f) {
        try {
            f();
        } catch (const MbdError& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    CHECK_THAT(error_of([&vh] { vh.set_steering(0.1); }), ContainsSubstring("MBD-V005"));
    kernel::System sys2;
    VehicleHandle rack_car = build_vehicle(sys2, VehicleTemplate::SportsCar());
    CHECK_THAT(error_of([&rack_car] { rack_car.set_steering_torque(1.0); }), ContainsSubstring("MBD-V005"));
}
