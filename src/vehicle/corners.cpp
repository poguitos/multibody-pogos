// The linkage corners of the vehicle template (plan task 7.6, decision D67):
// their mechanisms, the anti-roll bar on drop links, their motion at the
// design position, and the spring units set from it by virtual work.

#include <cmath>
#include <string>

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/vehicle/axes.hpp"
#include "mbd/vehicle/suspension/mass.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

namespace mbd {

namespace {

/// A left corner's offset as it is on the given side.
Vec3 side(const Vec3& offset, bool right) { return right ? detail::mirror_y(offset) : offset; }

kernel::Marker at(BodyIndex body, const Vec3& point) { return kernel::Marker{body, Transform3::FromTranslation(point)}; }

/// The mass one axle adds to the chassis's (TemplateVehicle::total_mass).
Real axle_mass_of(const AxleConfig& ax)
{
    Real m = 2.0 * ax.wheel_mass;
    switch (ax.suspension_type) {
        case SuspensionType::Simple: return m;
        case SuspensionType::DoubleWishbone:
            m += 2.0 * (2.0 * ax.arm_mass + ax.upright_mass);
            if (ax.spring_layout == SpringLayout::Pushrod) m += 2.0 * ax.rocker_mass;
            break;
        case SuspensionType::McPherson: m += 2.0 * (ax.arm_mass + ax.upright_mass); break;
        case SuspensionType::FiveLink: m += 2.0 * ax.upright_mass; break;
        case SuspensionType::TrailingArm: m += 2.0 * (ax.arm_mass + ax.upright_mass); break;
        case SuspensionType::SolidAxle: m += ax.axle_mass; break;
    }
    if (ax.arb_links.enabled) m += 2.0 * ax.arb_links.arm_mass;
    if (ax.is_steered) m += detail::steering_rack_mass;
    return m;
}

} // namespace

Real VehicleTemplate::total_mass() const
{
    Real lumped = 0.0;
    for (const LumpedMass& l : chassis.lumped_masses) lumped += l.mass;
    return chassis.mass + lumped + axle_mass_of(front_axle) + axle_mass_of(rear_axle);
}

// ============================================================================
// Hardpoints of a corner
// ============================================================================

MultilinkParams detail::make_five_link_params_for_corner(const Vec3& wc, const AxleConfig& ax, bool right)
{
    MultilinkParams p;
    p.wheel_center = wc;
    for (std::size_t i = 0; i < 5; ++i) {
        p.inner[i] = wc + side(ax.five_link.inner_offset[i], right);
        p.outer[i] = wc + side(ax.five_link.outer_offset[i], right);
    }
    p.upright_mass = ax.upright_mass;
    return p;
}

TrailingArmParams detail::make_trailing_arm_params_for_corner(const Vec3& wc, const AxleConfig& ax, bool right)
{
    TrailingArmParams p;
    p.wheel_center = wc;
    p.pivot = wc + side(ax.trailing_arm.pivot_offset, right);
    p.axis = side(ax.trailing_arm.axis, right);   // the mirrored hinge line
    p.arm_mass = ax.arm_mass;
    p.hub_mass = ax.upright_mass;
    return p;
}

SolidAxleParams detail::make_solid_axle_params(Real axle_x, const AxleConfig& ax)
{
    const SolidAxleHardpoints& h = ax.solid_axle;
    SolidAxleParams p;
    p.axle_center = Vec3(axle_x, 0.0, 0.0);
    p.half_track = ax.half_track;
    for (std::size_t s = 0; s < 2; ++s) {
        p.lower_chassis[s] = p.axle_center + side(h.lower_chassis_offset, s == 1);
        p.lower_axle[s] = p.axle_center + side(h.lower_axle_offset, s == 1);
    }
    p.upper_chassis = p.axle_center + h.upper_chassis_offset;
    p.upper_axle = p.axle_center + h.upper_axle_offset;
    p.panhard_chassis = p.axle_center + h.panhard_chassis_offset;
    p.panhard_axle = p.axle_center + h.panhard_axle_offset;
    p.axle_mass = ax.axle_mass;
    return p;
}

// ============================================================================
// Mechanisms
// ============================================================================

namespace {

void check_layout(const AxleConfig& ax, const char* where)
{
    MBD_THROW_IF(ax.spring_layout == SpringLayout::Pushrod && ax.suspension_type != SuspensionType::DoubleWishbone,
                 std::string("MBD-V002: ") + where + ": a pushrod (SpringLayout::Pushrod) is for a double wishbone");
}

} // namespace

CornerHandle detail::build_linkage_corner(kernel::System& sys, BodyIndex chassis, const Vec3& wc, const AxleConfig& ax,
                                          bool right, BodyIndex tierod_body, const std::string& name)
{
    using namespace kernel;
    check_layout(ax, "build_linkage_corner");
    CornerHandle ch;
    ch.type = ax.suspension_type;
    const char* lr = right ? "_R" : "_L";
    std::string wheel_name;
    switch (ax.suspension_type) {
        case SuspensionType::DoubleWishbone: {
            const auto p = make_dwb_params_for_corner(wc, ax.dwb, right, ax.arm_mass, ax.upright_mass);
            const auto dwb = build_double_wishbone_corner_dynamic(sys, chassis, p, tierod_body);
            ch.lca_body = dwb.lca_body;
            ch.uca_body = dwb.uca_body;
            ch.upright_body = dwb.upright_body;
            ch.tierod_constraint_idx = dwb.tierod_constraint_idx;
            ch.arb_link_body = dwb.lca_body;
            ch.arb_link_origin = p.lca_pivot;
            const DwbHardpoints& h = ax.dwb;
            if (ax.spring_layout == SpringLayout::CoilOver) {
                // From the chassis to the lower arm, whose frame is at its pivot.
                ch.spring_end1 = at(chassis, wc + side(h.spring_chassis_offset, right));
                ch.spring_end2 = at(dwb.lca_body, wc + side(h.spring_arm_offset, right) - p.lca_pivot);
            } else {
                // The rocker on a hinge on the chassis, its frame at its pivot
                // with the chassis's axes at the reference; the pushrod a
                // distance from the lower arm to it; the unit from the rocker
                // to the chassis.
                const Vec3 pivot = wc + side(h.rocker_pivot_offset, right);
                const Vec3 axis = side(h.rocker_axis, right);
                const Vec3 rod_arm = wc + side(h.pushrod_arm_offset, right);
                const Vec3 rod_rocker = wc + side(h.rocker_pushrod_offset, right);
                const Vec3 unit_rocker = wc + side(h.rocker_spring_offset, right);
                MBD_THROW_IF(!(axis.norm() > 0.0), "MBD-V002: build_linkage_corner: the rocker's axis is zero");
                const Mat3 R = detail::rotation_align_z_to(axis);
                ch.rocker_body = sys.model.add_body(
                    chassis, std::make_shared<RevoluteJointModel>(), Transform3(R, pivot), Transform3::FromRotation(R),
                    lumped_inertia(ax.rocker_mass, {Vec3::Zero(), rod_rocker - pivot, unit_rocker - pivot}),
                    name + lr + "_rocker");
                sys.constraints.push_back(std::make_shared<Distance>(at(dwb.lca_body, rod_arm - p.lca_pivot),
                                                                    at(ch.rocker_body, rod_rocker - pivot),
                                                                    (rod_rocker - rod_arm).norm()));
                ch.spring_end1 = at(chassis, wc + side(h.pushrod_spring_chassis_offset, right));
                ch.spring_end2 = at(ch.rocker_body, unit_rocker - pivot);
            }
            wheel_name = right ? "dwb_wheel_R" : "dwb_wheel_L";
            break;
        }
        case SuspensionType::McPherson: {
            const auto p = make_mcpherson_params_for_corner(wc, ax.mcpherson, right, ax.arm_mass, ax.upright_mass);
            const auto mc = build_mcpherson_corner_dynamic(sys, chassis, p, tierod_body);
            ch.lca_body = mc.lca_body;
            ch.upright_body = mc.upright_body;
            ch.tierod_constraint_idx = mc.tierod_constraint_idx;
            ch.arb_link_body = mc.lca_body;
            ch.arb_link_origin = p.lca_pivot;
            // Along the strut: from the top mount to its lower mount on the upright.
            ch.spring_end1 = at(chassis, p.strut_top_mount);
            ch.spring_end2 = at(mc.upright_body, p.strut_lower - p.wheel_center);
            wheel_name = right ? "mcpherson_wheel_R" : "mcpherson_wheel_L";
            break;
        }
        case SuspensionType::FiveLink: {
            const auto p = make_five_link_params_for_corner(wc, ax, right);
            const auto ml = build_multilink_corner_dynamic(sys, chassis, p, tierod_body);
            ch.upright_body = ml.upright_body;
            ch.tierod_constraint_idx = ml.link_constraint_indices[4];
            ch.arb_link_body = ml.upright_body;
            ch.arb_link_origin = wc;
            ch.spring_end1 = at(chassis, wc + side(ax.five_link.spring_chassis_offset, right));
            ch.spring_end2 = at(ml.upright_body, side(ax.five_link.spring_upright_offset, right));
            wheel_name = right ? "five_link_wheel_R" : "five_link_wheel_L";
            break;
        }
        case SuspensionType::TrailingArm: {
            MBD_THROW_IF(ax.is_steered, "MBD-V002: build_linkage_corner: a trailing arm cannot be steered");
            const auto p = make_trailing_arm_params_for_corner(wc, ax, right);
            const auto ta = build_trailing_arm_corner(sys, chassis, p);
            ch.upright_body = ta.arm_body;
            ch.arb_link_body = ta.arm_body;
            ch.arb_link_origin = wc;
            ch.spring_end1 = at(chassis, wc + side(ax.trailing_arm.spring_chassis_offset, right));
            ch.spring_end2 = at(ta.arm_body, side(ax.trailing_arm.spring_arm_offset, right));
            wheel_name = right ? "trailing_arm_wheel_R" : "trailing_arm_wheel_L";
            break;
        }
        default:
            MBD_THROW_IF(true, "MBD-V002: build_linkage_corner: a simple corner or a solid axle is not a linkage corner");
    }
    ch.wheel_body = add_wheel(sys.model, ch.upright_body, Vec3::Zero(), ax.wheel_mass, ax.tyre_body().free_radius,
                              wheel_name);
    ch.on_rack = tierod_body != chassis && ax.suspension_type != SuspensionType::TrailingArm;
    return ch;
}

std::array<CornerHandle, 2> detail::build_solid_axle_corners(kernel::System& sys, BodyIndex chassis, Real axle_x,
                                                             const AxleConfig& ax, const std::string& name)
{
    MBD_THROW_IF(ax.is_steered, "MBD-V002: build_solid_axle_corners: a solid axle cannot be steered");
    check_layout(ax, "build_solid_axle_corners");
    const SolidAxleParams p = make_solid_axle_params(axle_x, ax);
    const SolidAxle a = build_solid_axle(sys, chassis, p);
    std::array<CornerHandle, 2> out;
    for (std::size_t s = 0; s < 2; ++s) {
        const bool right = s == 1;
        CornerHandle& c = out[s];
        c.type = SuspensionType::SolidAxle;
        c.upright_body = a.axle_body;
        c.arb_link_body = a.axle_body;
        c.arb_link_origin = p.axle_center;
        c.wheel_center_in_upright = Vec3(0.0, right ? -p.half_track : p.half_track, 0.0);
        c.wheel_body = add_wheel(sys.model, a.axle_body, c.wheel_center_in_upright, ax.wheel_mass,
                                 ax.tyre_body().free_radius, name + (right ? "_wheel_R" : "_wheel_L"));
        c.spring_end1 = at(chassis, p.axle_center + side(ax.solid_axle.spring_chassis_offset, right));
        c.spring_end2 = at(a.axle_body, side(ax.solid_axle.spring_axle_offset, right));
    }
    return out;
}

void detail::build_anti_roll_bar_half(kernel::System& sys, BodyIndex chassis, const Vec3& left_wc,
                                      const AxleConfig& ax, CornerHandle& corner, bool right)
{
    using namespace kernel;
    const AntiRollBarLinks& b = ax.arb_links;
    auto point = [&](const Vec3& offset) { return right ? mirror_y(left_wc + offset) : Vec3(left_wc + offset); };
    const Vec3 bearing = point(b.bearing_offset);
    const Vec3 arm_end = point(b.arm_end_offset);
    const Vec3 lower = point(b.link_lower_offset);
    // Both halves turn about +Y (the joint's Z onto it), so that the bar's
    // twist is the difference of their angles.
    const Mat3 R = rotation_align_z_to(Vec3::UnitY());
    corner.arb_half_body = sys.model.add_body(chassis, std::make_shared<RevoluteJointModel>(), Transform3(R, bearing),
                                              Transform3::FromRotation(R),
                                              rod_inertia(b.arm_mass, Vec3::Zero(), arm_end - bearing),
                                              right ? "arb_half_R" : "arb_half_L");
    sys.constraints.push_back(std::make_shared<Distance>(at(corner.arb_half_body, arm_end - bearing),
                                                        at(corner.arb_link_body, lower - corner.arb_link_origin),
                                                        (lower - arm_end).norm()));
}

// ============================================================================
// The corner alone, and its motion at the design position
// ============================================================================

CornerRig build_corner_rig(kernel::System& sys, const VehicleTemplate& tmpl, int corner_idx)
{
    using namespace kernel;
    MBD_THROW_IF(corner_idx < 0 || corner_idx > 3, "MBD-V002: build_corner_rig: corners are 0 to 3");
    const int axle = corner_idx / 2;
    const AxleConfig& ax = axle == 0 ? tmpl.front_axle : tmpl.rear_axle;
    MBD_THROW_IF(ax.suspension_type == SuspensionType::Simple,
                 "MBD-V002: build_corner_rig: corner " + std::to_string(corner_idx)
                     + " is a simple corner: it has no linkage");
    const Real x = axle == 0 ? tmpl.front_axle_x : -tmpl.rear_axle_x;
    const Vec3 left_wc(x, ax.half_track, 0.0);

    sys.model.gravity = iso_gravity();
    CornerRig rig;
    rig.chassis_body = sys.model.add_body(0, std::make_shared<FixedJointModel>(), Transform3::Identity(),
                                          Transform3::Identity(),
                                          RigidBodyInertia::from_solid_box(10000.0, tmpl.chassis.half_extents),
                                          "rig_chassis");
    std::vector<bool> right_side;
    if (ax.suspension_type == SuspensionType::SolidAxle) {
        const auto both = detail::build_solid_axle_corners(sys, rig.chassis_body, x, ax, "rig_axle");
        rig.corners = {both[0], both[1]};
        right_side = {false, true};
    } else {
        const bool right = corner_idx % 2 == 1;
        const Vec3 wc = right ? detail::mirror_y(left_wc) : left_wc;
        rig.corners = {detail::build_linkage_corner(sys, rig.chassis_body, wc, ax, right, rig.chassis_body, "rig")};
        right_side = {right};
    }
    for (std::size_t i = 0; i < rig.corners.size(); ++i) {
        CornerHandle& c = rig.corners[i];
        if (ax.arb_links.enabled) detail::build_anti_roll_bar_half(sys, rig.chassis_body, left_wc, ax, c, right_side[i]);
        rig.drivers.push_back(sys.constraints.size());
        sys.constraints.push_back(point_height_driver(c.upright_body, c.wheel_center_in_upright, 2, 0.0));
    }
    return rig;
}

detail::CornerMotion detail::corner_motion(const VehicleTemplate& tmpl, int axle, Real loaded_radius)
{
    kernel::System sys;
    const CornerRig rig = build_corner_rig(sys, tmpl, axle == 0 ? 0 : 2);
    Kinematics k(sys);
    MBD_THROW_IF(!k.solve(50, 1e-12),
                 "MBD-V003: corner_motion: the corner's mechanism does not close at its design position");
    const VecX v = k.velocities();
    kernel::Data data(sys.model);
    kernel::forward_kinematics(sys.model, data, k.q, v);
    auto point = [&](const kernel::Marker& m) {
        const RigidBodyState s = kernel::body_state(data, m.body);
        return Vec3(s.p_WB + s.q_WB * m.X_BM.p);
    };
    auto rate = [&](const kernel::Marker& m) {
        const RigidBodyState s = kernel::body_state(data, m.body);
        return Vec3(s.v_WB + s.w_WB.cross(s.q_WB * m.X_BM.p));
    };

    CornerMotion out;
    out.sides = static_cast<int>(rig.corners.size());
    Real springs = 0.0;
    for (std::size_t i = 0; i < rig.corners.size(); ++i) {
        const CornerHandle& c = rig.corners[i];
        const Vec3 e = point(c.spring_end2) - point(c.spring_end1);
        out.spring_length[i] = e.norm();
        out.spring_rate[i] = (rate(c.spring_end2) - rate(c.spring_end1)).dot(e / e.norm());
        springs += out.spring_rate[i];
        // The tyre's contact: the wheel circle's lowest point, loaded_radius
        // below the centre within the wheel plane, carried by the upright.
        const RigidBodyState u = kernel::body_state(data, c.upright_body);
        const Vec3 centre = u.p_WB + u.q_WB * c.wheel_center_in_upright;
        const Vec3 axis = u.q_WB * Vec3::UnitY();
        const Vec3 down = (-Vec3::UnitZ() + Vec3::UnitZ().dot(axis) * axis).normalized();
        const Vec3 contact = centre + loaded_radius * down;
        out.contact_rate[i] = (u.v_WB + u.w_WB.cross(contact - u.p_WB)).z();
        const Vec3 heading = Vec3(u.q_WB * Vec3::UnitX()).cwiseProduct(Vec3(1.0, 1.0, 0.0)).normalized();
        out.camber_rate[i] = u.w_WB.dot(heading);
        if (c.arb_half_body > 0) out.bar_rate[i] = kernel::body_state(data, c.arb_half_body).w_WB.y();
    }
    for (int b = 1; b < sys.model.nbodies(); ++b) {
        if (b == rig.chassis_body) continue;
        const RigidBodyInertia& I = sys.model.inertia[static_cast<std::size_t>(b)];
        out.weight_rate += I.mass * rate(kernel::Marker{b, Transform3::FromTranslation(I.com_B)}).z();
    }
    MBD_THROW_IF(!(std::abs(springs) > 1e-6),
                 "MBD-V003: corner_motion: the spring unit does not move with the wheel (motion ratio "
                     + std::to_string(springs) + ")");
    return out;
}

Real detail::design_spring_force(const CornerMotion& m, Real tyre_load, Real overturning)
{
    Real springs = 0.0, contacts = 0.0;
    for (int i = 0; i < m.sides; ++i) {
        springs += m.spring_rate[static_cast<std::size_t>(i)];
        contacts += m.contact_rate[static_cast<std::size_t>(i)];
    }
    const Real moments = overturning * (m.camber_rate[0] - (m.sides == 2 ? m.camber_rate[1] : 0.0));
    return (g_accel * m.weight_rate - tyre_load * contacts - moments) / springs;
}

void detail::add_spring_unit(kernel::System& sys, const AxleConfig& ax, CornerHandle& c, Real design_force,
                             Real design_length)
{
    MBD_THROW_IF(!(ax.k_spring > 0.0), "MBD-V004: add_spring_unit: a linkage corner's spring needs k_spring > 0");
    const BodyIndex b1 = c.spring_end1.body, b2 = c.spring_end2.body;
    const Vec3 a1 = c.spring_end1.X_BM.p, a2 = c.spring_end2.X_BM.p;
    const Curve damper = ax.damper_curve.is_zero() ? Curve::linear(ax.c_damper) : ax.damper_curve;
    const bool top_mount = ax.top_mount_stiffness > 0.0;

    SpringDamperParams p;
    p.spring = Curve::linear(ax.k_spring);
    p.free_length = design_length + p.spring.inverse(design_force);
    if (!top_mount) p.damper = damper;
    auto unit = std::make_shared<SpringDamper>(b1, b2, a1, a2, p);
    c.spring = unit.get();
    sys.force_elements.push_back(std::move(unit));

    if (top_mount) {
        MaxwellDamperParams m;
        m.damper = damper;
        m.series = Curve::linear(ax.top_mount_stiffness);
        auto maxwell = std::make_shared<MaxwellDamper>(b1, b2, a1, a2, m);
        c.top_mount = maxwell.get();
        sys.force_elements.push_back(std::move(maxwell));
    }
    if (std::isfinite(ax.bump_travel) || std::isfinite(ax.rebound_travel)) {
        // The stops measured from the design length: their own free length.
        SpringDamperParams s;
        s.free_length = design_length;
        s.bump_clearance = ax.bump_travel;
        s.bump_stop = ax.bump_stop;
        s.rebound_clearance = ax.rebound_travel;
        s.rebound_stop = ax.rebound_stop;
        auto stops = std::make_shared<SpringDamper>(b1, b2, a1, a2, s);
        c.stops = stops.get();
        sys.force_elements.push_back(std::move(stops));
    }
    c.design_spring_force = design_force;
    c.design_spring_length = design_length;
}

} // namespace mbd
