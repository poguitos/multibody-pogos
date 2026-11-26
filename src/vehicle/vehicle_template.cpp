#include "mbd/vehicle/vehicle_template.hpp"

#include <algorithm>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/vehicle/axes.hpp"

namespace mbd {

VehicleTemplate VehicleTemplate::DefaultSedan()
{
    VehicleTemplate t;
    t.name = "default_sedan";

    t.chassis.mass = 1400.0;
    t.front_axle_x = 1.35;
    t.rear_axle_x  = 1.35;

    t.front_axle.suspension_type = SuspensionType::Simple;
    t.front_axle.is_steered = true;
    t.front_axle.half_track = 0.8;
    t.front_axle.k_spring   = 25000.0;
    t.front_axle.c_damper   = 2000.0;

    t.rear_axle.suspension_type = SuspensionType::Simple;
    t.rear_axle.is_steered = false;
    t.rear_axle.half_track = 0.8;
    t.rear_axle.k_spring   = 22000.0;
    t.rear_axle.c_damper   = 1800.0;

    t.drivetrain.layout = DriveLayout::RWD;

    return t;
}

VehicleTemplate VehicleTemplate::SportsCar()
{
    VehicleTemplate t;
    t.name = "sports_car";

    t.chassis.mass = 1200.0;
    t.chassis.half_extents = Vec3(1.4, 0.75, 0.25);

    t.front_axle_x = 1.25;
    t.rear_axle_x  = 1.45;

    t.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    t.front_axle.is_steered = true;
    t.front_axle.half_track = 0.78;
    t.front_axle.k_spring   = 35000.0;
    t.front_axle.c_damper   = 2500.0;
    t.front_axle.wheel_mass = 35.0;

    t.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    t.rear_axle.is_steered = false;
    t.rear_axle.half_track = 0.78;
    t.rear_axle.k_spring   = 40000.0;
    t.rear_axle.c_damper   = 2800.0;
    t.rear_axle.wheel_mass = 35.0;

    t.drivetrain.layout = DriveLayout::RWD;
    t.drivetrain.engine.max_torque = 450.0;

    return t;
}

VehicleTemplate VehicleTemplate::FWDHatchback()
{
    VehicleTemplate t;
    t.name = "fwd_hatchback";

    t.chassis.mass = 1100.0;
    t.chassis.half_extents = Vec3(1.2, 0.72, 0.28);

    t.front_axle_x = 1.0;
    t.rear_axle_x  = 1.5;

    t.front_axle.suspension_type = SuspensionType::McPherson;
    t.front_axle.is_steered = true;
    t.front_axle.half_track = 0.76;
    t.front_axle.k_spring   = 22000.0;
    t.front_axle.c_damper   = 1800.0;

    t.rear_axle.suspension_type = SuspensionType::Simple;
    t.rear_axle.is_steered = false;
    t.rear_axle.half_track = 0.74;
    t.rear_axle.k_spring   = 20000.0;
    t.rear_axle.c_damper   = 1600.0;

    t.drivetrain.layout = DriveLayout::FWD;
    t.drivetrain.engine.max_torque = 250.0;

    return t;
}

void VehicleHandle::set_steering(Real delta)
{
    const bool straight = std::abs(delta) < Real(1e-10);
    const Real L = tmpl.wheelbase();
    const Real R = straight ? Real(0.0) : L / std::tan(delta);

    for (int axle = 0; axle < 2; ++axle) {
        const AxleConfig& ax = (axle == 0) ? tmpl.front_axle : tmpl.rear_axle;
        Real ratio_sum = 0.0;
        int n_on_rack = 0;
        for (int side = 0; side < 2; ++side) {
            CornerHandle& ch = corners[2 * axle + side];
            if (ch.type == SuspensionType::Simple) {
                Real angle = 0.0;
                if (!straight && ax.is_steered) {
                    const Real ht = ax.half_track;
                    angle = (side == 0) ? std::atan(L / (R - ht)) : std::atan(L / (R + ht));
                }
                if (ch.tire) ch.tire->steer_angle = angle;
            } else {
                if (ch.tire) ch.tire->steer_angle = 0.0;
                if (ch.on_rack && ch.rack_per_rad != 0.0) {
                    ratio_sum += std::abs(ch.rack_per_rad);
                    ++n_on_rack;
                }
            }
        }
        if (rack[axle]) {
            rack[axle]->travel = (!straight && ax.is_steered && n_on_rack > 0)
                               ? delta * ratio_sum / n_on_rack
                               : Real(0.0);
        }
    }
}

void VehicleHandle::set_road(const std::shared_ptr<const Road>& road)
{
    MBD_THROW_IF(!road, "MBD-F051: VehicleHandle::set_road: the road is null");
    for (auto& corner : corners) {
        if (corner.tire) corner.tire->road = road;
    }
}

std::pair<AntiRollBar*, AntiRollBar*> VehicleHandle::install_anti_roll_bars(
    kernel::System& sys,
    const std::vector<RigidBodyState>& equilibrium_states)
{
    AntiRollBar* front_arb = nullptr;
    AntiRollBar* rear_arb  = nullptr;

    if (tmpl.front_axle.k_arb > 0.0 && !detail::anti_roll_bar_on_links(tmpl.front_axle)) {
        auto arb = std::make_shared<AntiRollBar>(
            chassis_body,
            corners[0].upright_body,  // FL
            corners[1].upright_body,  // FR
            tmpl.front_axle.k_arb,
            tmpl.front_axle.c_arb);
        arb->capture_reference(equilibrium_states);
        front_arb = arb.get();
        sys.force_elements.push_back(std::move(arb));
    }

    if (tmpl.rear_axle.k_arb > 0.0 && !detail::anti_roll_bar_on_links(tmpl.rear_axle)) {
        auto arb = std::make_shared<AntiRollBar>(
            chassis_body,
            corners[2].upright_body,  // RL
            corners[3].upright_body,  // RR
            tmpl.rear_axle.k_arb,
            tmpl.rear_axle.c_arb);
        arb->capture_reference(equilibrium_states);
        rear_arb = arb.get();
        sys.force_elements.push_back(std::move(arb));
    }

    return {front_arb, rear_arb};
}

AerodynamicForce* VehicleHandle::install_aerodynamics(kernel::System& sys)
{
    if (tmpl.chassis.CdA <= 0.0 && tmpl.chassis.ClA <= 0.0) {
        return nullptr;
    }

    AeroParams p;
    p.CdA = tmpl.chassis.CdA;
    p.ClA = tmpl.chassis.ClA;
    p.cop_offset_chassis = tmpl.chassis.aero_cop_offset;
    p.h_ref = tmpl.chassis.aero_h_ref;
    p.dClA_dh = tmpl.chassis.aero_dClA_dh;
    p.ground_normal_W = ground_up(sys.model.gravity);

    auto aero = std::make_shared<AerodynamicForce>(chassis_body, p);
    AerodynamicForce* ptr = aero.get();
    sys.force_elements.push_back(std::move(aero));
    return ptr;
}

CornerHandle detail::build_simple_corner(
    kernel::System& sys,
    BodyIndex chassis_body,
    const Vec3& mount_pos_chassis,
    const AxleConfig& axle,
    const std::string& name)
{
    CornerHandle ch;
    ch.type = SuspensionType::Simple;

    // Prismatic joint: chassis → slider, axis = chassis -Z (downward): Rx(pi)
    // maps the joint's Z to -Z, so a positive coordinate moves the wheel down.
    // The slider is massless: the unsprung mass is the wheel's.
    const Mat3 R_susp = Eigen::AngleAxisd(pi, Vec3::UnitX()).toRotationMatrix();
    ch.upright_body = sys.model.add_body(
        chassis_body, std::make_shared<kernel::PrismaticJointModel>(),
        Transform3(R_susp, mount_pos_chassis), Transform3::FromRotation(R_susp),
        RigidBodyInertia{}, name + "_carrier");
    ch.wheel_body = add_wheel(sys.model, ch.upright_body, Vec3::Zero(), axle.wheel_mass,
                              axle.tyre_body().free_radius, name);

    // Spring-damper
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        chassis_body, ch.upright_body,
        mount_pos_chassis, Vec3::Zero(),
        axle.k_spring, axle.c_damper, axle.spring_rest_length));

    // Tyre
    auto tire = std::make_shared<FullTireForce>(ch.wheel_body, axle.make_tyre());
    ch.tire = tire.get();
    sys.force_elements.push_back(std::move(tire));

    return ch;
}

DoubleWishboneParams detail::make_dwb_params_for_corner(
    const Vec3& wheel_center_chassis,
    const DwbHardpoints& hp,
    bool is_right_side,
    Real arm_mass,
    Real upright_mass)
{
    auto offset = [&](const Vec3& v) -> Vec3 {
        Vec3 world = wheel_center_chassis + (is_right_side ? mirror_y(v) : v);
        return world;
    };

    DoubleWishboneParams p;
    p.wheel_center   = wheel_center_chassis;
    p.lca_pivot      = offset(hp.lca_pivot_offset);
    p.lca_outer      = offset(hp.lca_outer_offset);
    p.uca_pivot      = offset(hp.uca_pivot_offset);
    p.uca_outer      = offset(hp.uca_outer_offset);
    p.tierod_inner   = offset(hp.tierod_inner_offset);
    p.tierod_outer   = offset(hp.tierod_outer_offset);
    p.arm_axis       = hp.arm_axis;
    p.arm_mass       = arm_mass;
    p.upright_mass   = upright_mass;
    return p;
}

McPhersonParams detail::make_mcpherson_params_for_corner(
    const Vec3& wheel_center_chassis,
    const McPhersonHardpoints& hp,
    bool is_right_side,
    Real arm_mass,
    Real upright_mass)
{
    auto offset = [&](const Vec3& v) -> Vec3 {
        Vec3 world = wheel_center_chassis + (is_right_side ? mirror_y(v) : v);
        return world;
    };

    McPhersonParams p;
    p.wheel_center   = wheel_center_chassis;
    p.lca_pivot      = offset(hp.lca_pivot_offset);
    p.lca_outer      = offset(hp.lca_outer_offset);
    p.strut_top_mount = offset(hp.strut_top_offset);
    p.strut_lower     = offset(hp.strut_lower_offset);
    p.tierod_inner    = offset(hp.tierod_inner_offset);
    p.tierod_outer    = offset(hp.tierod_outer_offset);
    p.arm_axis        = hp.arm_axis;
    p.arm_mass        = arm_mass;
    p.upright_mass    = upright_mass;
    return p;
}

BodyIndex detail::add_steering_rack(kernel::System& sys, BodyIndex chassis_body,
                                    const std::shared_ptr<SteeringRack>& rack,
                                    const std::string& name)
{
    // Rx(-pi/2) maps the joint's Z (the sliding axis) to the chassis's Y; the
    // same rotation on both sides keeps the rack's frame along the chassis's.
    const Mat3 R_rack = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    const BodyIndex body = sys.model.add_body(
        chassis_body, std::make_shared<kernel::PrismaticJointModel>(),
        Transform3::FromRotation(R_rack), Transform3::FromRotation(R_rack),
        RigidBodyInertia::from_solid_box(steering_rack_mass, Vec3(0.02, 0.4, 0.02)), name);
    sys.constraints.push_back(std::make_shared<kernel::JointDriver>(
        sys.model, body,
        kernel::TimeFunction([rack](Real) { return rack->travel; },
                             [](Real) { return 0.0; },
                             [](Real) { return 0.0; })));
    return body;
}

Real detail::calibrate_rack(const VehicleTemplate& tmpl, const AxleConfig& ax,
                            const Vec3& wheel_center_chassis, bool is_right)
{
    if (ax.suspension_type != SuspensionType::DoubleWishbone && ax.suspension_type != SuspensionType::McPherson
        && ax.suspension_type != SuspensionType::FiveLink) {
        return 0.0;
    }
    kernel::System calib;
    const BodyIndex chassis = calib.model.add_body(
        0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
        Transform3::Identity(),
        RigidBodyInertia::from_solid_box(10000.0, tmpl.chassis.half_extents), "calib_chassis");
    auto rack = std::make_shared<SteeringRack>();
    const BodyIndex rack_body = add_steering_rack(calib, chassis, rack, "calib_rack");
    const BodyIndex upright =
        build_linkage_corner(calib, chassis, wheel_center_chassis, ax, is_right, rack_body, "calib").upright_body;

    Kinematics k(calib);
    if (!k.solve(50, 1e-10)) return 0.0;
    const Real toe_ref = extract_toe(k.state(upright));

    const Real rack_step = 0.005;   // +5 mm toward +Y (left)
    rack->travel = rack_step;
    if (!k.solve(50, 1e-10)) return 0.0;
    const Real dtoe = extract_toe(k.state(upright)) - toe_ref;
    return std::abs(dtoe) > 1e-6 ? rack_step / dtoe : Real(0.0);
}

VehicleHandle build_vehicle(kernel::System& sys,
                            const VehicleTemplate& tmpl)
{
    VehicleHandle vh;
    vh.tmpl = tmpl;
    sys.model.gravity = iso_gravity();

    // --- Chassis (body 1) ---
    vh.chassis_body = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(),
        Transform3::Identity(), Transform3::Identity(),
        RigidBodyInertia::from_solid_box(tmpl.chassis.mass, tmpl.chassis.half_extents),
        tmpl.name + "_chassis");

    // --- Steering racks ---
    const std::array<const AxleConfig*, 2> axles{{&tmpl.front_axle, &tmpl.rear_axle}};
    for (int a = 0; a < 2; ++a) {
        const AxleConfig& ax = *axles[static_cast<std::size_t>(a)];
        if (ax.is_steered && ax.suspension_type != SuspensionType::Simple) {
            vh.rack[a] = std::make_shared<SteeringRack>();
            vh.rack_body[a] = detail::add_steering_rack(sys, vh.chassis_body, vh.rack[a],
                                                        a == 0 ? "rack_front" : "rack_rear");
        }
    }

    // --- Corner positions in chassis frame ---
    struct CornerDef {
        Vec3 mount_pos;
        const AxleConfig& axle;
        bool is_right;
        std::string name;
    };

    const std::array<CornerDef, 4> corner_defs = {{
        {Vec3( tmpl.front_axle_x,  tmpl.front_axle.half_track, 0.0), tmpl.front_axle, false, "FL"},
        {Vec3( tmpl.front_axle_x, -tmpl.front_axle.half_track, 0.0), tmpl.front_axle, true,  "FR"},
        {Vec3(-tmpl.rear_axle_x,   tmpl.rear_axle.half_track,  0.0), tmpl.rear_axle,  false, "RL"},
        {Vec3(-tmpl.rear_axle_x,  -tmpl.rear_axle.half_track,  0.0), tmpl.rear_axle,  true,  "RR"},
    }};

    const bool linkage = tmpl.front_axle.suspension_type != SuspensionType::Simple
                      || tmpl.rear_axle.suspension_type != SuspensionType::Simple;

    // --- A car of simple corners only: as before task 7.6, each corner
    // whole (slider, wheel, spring, tyre), its mount at the chassis frame's
    // Z = 0.
    if (!linkage) {
        for (int c = 0; c < 4; ++c) {
            const auto& cd = corner_defs[static_cast<std::size_t>(c)];
            vh.corners[c] = detail::build_simple_corner(sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.name);
        }
    }

    // --- The linkage axles' mechanisms (task 7.6): a linkage corner's or a
    // solid axle's bodies, joints and constraints, and an anti-roll bar's
    // halves on drop links; their wheel centres at the chassis frame's Z = 0
    // at the design position.
    for (int a = 0; a < 2; ++a) {
        const AxleConfig& ax = *axles[static_cast<std::size_t>(a)];
        if (ax.suspension_type == SuspensionType::Simple) continue;
        const BodyIndex rack_body = vh.rack_body[static_cast<std::size_t>(a)];
        const BodyIndex tierod_body = rack_body > 0 ? rack_body : vh.chassis_body;
        if (ax.suspension_type == SuspensionType::SolidAxle) {
            const Real x = a == 0 ? tmpl.front_axle_x : -tmpl.rear_axle_x;
            const auto both = detail::build_solid_axle_corners(sys, vh.chassis_body, x, ax,
                                                               a == 0 ? "front_axle" : "rear_axle");
            vh.corners[2 * a] = both[0];
            vh.corners[2 * a + 1] = both[1];
        } else {
            for (int s = 0; s < 2; ++s) {
                const auto& cd = corner_defs[static_cast<std::size_t>(2 * a + s)];
                vh.corners[2 * a + s] = detail::build_linkage_corner(sys, vh.chassis_body, cd.mount_pos, ax,
                                                                     cd.is_right, tierod_body, cd.name);
            }
        }
        if (ax.arb_links.enabled) {
            for (int s = 0; s < 2; ++s) {
                detail::build_anti_roll_bar_half(sys, vh.chassis_body, corner_defs[static_cast<std::size_t>(2 * a)].mount_pos,
                                                 ax, vh.corners[2 * a + s], s == 1);
            }
        }
    }

    // --- Each axle's tyre load at rest, from the masses made and where they
    // lie (the reference configuration, the chassis at the origin): the
    // moments of the weights about the rear contacts' line give the front's.
    // In a mixed car the simple corners are not made yet: their wheels (all
    // their mass, the slider having none) are at their axle's X.
    {
        kernel::Data data(sys.model);
        kernel::forward_kinematics(sys.model, data, sys.model.neutral_configuration());
        const Real x_f = tmpl.front_axle_x, x_r = -tmpl.rear_axle_x;
        Real mass = 0.0, moment = 0.0;
        for (int b = 1; b < sys.model.nbodies(); ++b) {
            const RigidBodyInertia& I = sys.model.inertia[static_cast<std::size_t>(b)];
            const RigidBodyState s = kernel::body_state(data, b);
            mass += I.mass;
            moment += I.mass * ((s.p_WB + s.q_WB * I.com_B).x() - x_r);
        }
        if (linkage) {
            for (int a = 0; a < 2; ++a) {
                const AxleConfig& ax = *axles[static_cast<std::size_t>(a)];
                if (ax.suspension_type != SuspensionType::Simple) continue;
                mass += 2.0 * ax.wheel_mass;
                moment += 2.0 * ax.wheel_mass * ((a == 0 ? x_f : x_r) - x_r);
            }
        }
        const Real front = 0.5 * g_accel * moment / (x_f - x_r);
        vh.design_tyre_load = {{front, 0.5 * g_accel * mass - front}};
    }

    // --- A mixed car's simple corners: each mount raised by its spring's
    // length under its load less its wheel's weight, so that at the design
    // position its wheel centre too is at the chassis frame's Z = 0.
    if (linkage) {
        for (int c = 0; c < 4; ++c) {
            const auto& cd = corner_defs[static_cast<std::size_t>(c)];
            const AxleConfig& ax = cd.axle;
            if (ax.suspension_type != SuspensionType::Simple) continue;
            const Real force = vh.design_tyre_load[static_cast<std::size_t>(c / 2)] - ax.wheel_mass * g_accel;
            const Real length = ax.spring_rest_length - force / ax.k_spring;
            vh.corners[c] = detail::build_simple_corner(sys, vh.chassis_body, cd.mount_pos + Vec3(0.0, 0.0, length),
                                                        ax, cd.name);
            vh.corners[c].design_spring_force = force;
            vh.corners[c].design_spring_length = length;
        }
    }

    // --- The linkage corners' spring units, set by virtual work to hold the
    // design position under those loads; an anti-roll bar's torsion on drop
    // links, set to give k_arb and c_arb at the wheels.
    for (int a = 0; a < 2; ++a) {
        const AxleConfig& ax = *axles[static_cast<std::size_t>(a)];
        if (ax.suspension_type == SuspensionType::Simple) continue;
        const Real load = vh.design_tyre_load[static_cast<std::size_t>(a)];
        const TyreBody tyre = ax.tyre_body();
        const detail::CornerMotion m = detail::corner_motion(tmpl, a, tyre.free_radius - load / tyre.vertical_stiffness);
        // The left tyre's overturning moment at rest, as the tyre on its
        // wheel evaluates it there (no slip, no inclination, no speed),
        // mirrored when its parameters describe a right tyre.
        TyreInput at_rest;
        at_rest.Fz = load;
        at_rest.Vcx = 0.0;
        const Real overturning = (tyre.side == TyreSide::Right ? -1.0 : 1.0) * ax.make_tyre()->evaluate(at_rest).Mx;
        const Real force = detail::design_spring_force(m, load, overturning);
        for (int s = 0; s < 2; ++s) {
            const std::size_t k = static_cast<std::size_t>(std::min(s, m.sides - 1));
            detail::add_spring_unit(sys, ax, vh.corners[2 * a + s], force, m.spring_length[k]);
        }
        if (ax.arb_links.enabled && (ax.k_arb > 0.0 || ax.c_arb > 0.0)) {
            const Real r = m.bar_rate[0];
            MBD_THROW_IF(!(std::abs(r) > 1e-9),
                         "MBD-V003: build_vehicle: the anti-roll bar on drop links does not turn with the wheel");
            RotationalSpringDamperParams tp;
            tp.spring = Curve::linear(ax.k_arb / (r * r));
            tp.damper = Curve::linear(ax.c_arb / (r * r));
            const Transform3 along_y = Transform3::FromRotation(detail::rotation_align_z_to(Vec3::UnitY()));
            auto torsion = std::make_shared<RotationalSpringDamper>(vh.corners[2 * a].arb_half_body, along_y,
                                                                    vh.corners[2 * a + 1].arb_half_body, along_y, tp);
            vh.arb_torsion[static_cast<std::size_t>(a)] = torsion.get();
            sys.force_elements.push_back(std::move(torsion));
        }
    }

    // --- Tyres of the linkage corners (a simple corner has its own) ---
    for (int c = 0; c < 4; ++c) {
        CornerHandle& corner = vh.corners[static_cast<std::size_t>(c)];
        if (corner.type != SuspensionType::Simple) {
            const AxleConfig& ax = *axles[static_cast<std::size_t>(c / 2)];
            auto tire = std::make_shared<FullTireForce>(corner.wheel_body, ax.make_tyre());
            corner.tire = tire.get();
            sys.force_elements.push_back(std::move(tire));
        }
        corner.tire->mount(corner_defs[static_cast<std::size_t>(c)].is_right);
    }

    // --- Calibrate rack travel per radian of toe, corner by corner ---
    for (int c = 0; c < 4; ++c) {
        auto& corner = vh.corners[c];
        if (!corner.on_rack) continue;
        const auto& cd = corner_defs[static_cast<std::size_t>(c)];
        corner.rack_per_rad = detail::calibrate_rack(tmpl, cd.axle, cd.mount_pos, cd.is_right);
    }

    return vh;
}

void roll_wheels(kernel::Simulator& sim, const VehicleHandle& vh)
{
    for (const CornerHandle& corner : vh.corners) {
        if (corner.tire) set_free_rolling(sim, corner.wheel_body, *corner.tire);
    }
}

namespace {

/// The placement of a car of simple corners only, as it was before task 7.6
/// (see set_vehicle_equilibrium).
void legacy_vehicle_equilibrium(kernel::Simulator& sim, const VehicleHandle& vh)
{
    const auto& t = vh.tmpl;
    const kernel::Model& model = sim.system.model;

    sim.q = model.neutral_configuration();
    sim.v.setZero();

    // Per-axle static load (from CG position)
    const Real L = t.wheelbase();
    const Real W_total = t.total_mass() * g_accel;
    const Real W_front_per = W_total * t.rear_axle_x / L * 0.5;
    const Real W_rear_per  = W_total * t.front_axle_x / L * 0.5;

    // Per-corner world-frame wheel center height (contact at z=0)
    const Real wheel_z_front_world =
        t.front_axle.tyre_body().free_radius - W_front_per / t.front_axle.tyre_body().vertical_stiffness;
    const Real wheel_z_rear_world =
        t.rear_axle.tyre_body().free_radius - W_rear_per / t.rear_axle.tyre_body().vertical_stiffness;

    auto chassis_height = [](const CornerHandle& corner, const AxleConfig& ax, Real wheel_z, Real W_per) {
        if (corner.type == SuspensionType::Simple) {
            const Real spring_compr = W_per / ax.k_spring;
            return wheel_z + (ax.spring_rest_length - spring_compr);
        }
        return wheel_z;
    };
    const Real chassis_z_front = chassis_height(vh.corners[0], t.front_axle, wheel_z_front_world, W_front_per);
    const Real chassis_z_rear  = chassis_height(vh.corners[2], t.rear_axle, wheel_z_rear_world, W_rear_per);
    const Real chassis_z = 0.5 * (chassis_z_front + chassis_z_rear);

    // The free joint's coordinates start with the translation (x, y, z).
    sim.q(model.idx_q[vh.chassis_body] + 2) = chassis_z;

    // Simple corners: the prismatic coordinate is the suspension travel.
    for (int c = 0; c < 4; ++c) {
        if (vh.corners[c].type != SuspensionType::Simple) continue;
        const Real wheel_z = (c < 2) ? wheel_z_front_world : wheel_z_rear_world;
        sim.q(model.idx_q[vh.corners[c].upright_body]) = chassis_z - wheel_z;
    }

    sim.refresh();
}

} // namespace

void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleHandle& vh)
{
    const auto& t = vh.tmpl;
    const bool linkage = t.front_axle.suspension_type != SuspensionType::Simple
                      || t.rear_axle.suspension_type != SuspensionType::Simple;
    if (!linkage) {
        legacy_vehicle_equilibrium(sim, vh);
        return;
    }
    const kernel::Model& model = sim.system.model;
    sim.q = model.neutral_configuration();
    sim.v.setZero();

    // Each axle's loaded radius. At the design position every wheel centre
    // is at the chassis frame's Z = 0 (build_vehicle raised a mixed car's
    // simple corners' mounts to put them there).
    std::array<Real, 2> radius{};
    for (std::size_t a = 0; a < 2; ++a) {
        const TyreBody tyre = (a == 0 ? t.front_axle : t.rear_axle).tyre_body();
        radius[a] = tyre.free_radius - vh.design_tyre_load[a] / tyre.vertical_stiffness;
    }
    // The chassis's origin at height h, pitched by theta about Y: its point
    // (x, y, 0) is at h - x sin(theta). Each axle's wheel centres must be at
    // their loaded radius: h - x_a sin(theta) = radius_a.
    const Real x_f = t.front_axle_x, x_r = -t.rear_axle_x;
    const Real theta = std::asin((radius[1] - radius[0]) / (x_f - x_r));
    const Real h = radius[0] + x_f * std::sin(theta);
    const int cq = model.idx_q[static_cast<std::size_t>(vh.chassis_body)];
    sim.q(cq + 2) = h;
    // The free joint's quaternion (x, y, z, w): a turn by theta about Y.
    sim.q(cq + 3) = 0.0;
    sim.q(cq + 4) = std::sin(0.5 * theta);
    sim.q(cq + 5) = 0.0;
    sim.q(cq + 6) = std::cos(0.5 * theta);
    // A simple corner's slider: the distance from its mount down to its
    // wheel centre, its spring's length at the design position.
    for (const CornerHandle& corner : vh.corners) {
        if (corner.type != SuspensionType::Simple) continue;
        sim.q(model.idx_q[static_cast<std::size_t>(corner.upright_body)]) = corner.design_spring_length;
    }
    sim.refresh();
}

std::pair<DoubleWishboneCorner, std::size_t> build_dwb_for_analysis(
    kernel::System& sys,
    const VehicleTemplate& tmpl,
    int corner_idx)
{
    const auto& ax = (corner_idx < 2) ? tmpl.front_axle : tmpl.rear_axle;
    const bool is_right = (corner_idx % 2 == 1);
    const Real axle_x = (corner_idx < 2) ? tmpl.front_axle_x : -tmpl.rear_axle_x;
    const Real track_y = is_right ? -ax.half_track : ax.half_track;

    const Vec3 wheel_center(axle_x, track_y, 0.25);

    auto p = detail::make_dwb_params_for_corner(
        wheel_center, ax.dwb, is_right, ax.arm_mass, ax.upright_mass);

    auto dwb = build_double_wishbone_corner(sys, p);
    return {dwb, dwb.bump_constraint_idx};
}

std::pair<McPhersonCorner, std::size_t> build_mcpherson_for_analysis(
    kernel::System& sys,
    const VehicleTemplate& tmpl,
    int corner_idx)
{
    const auto& ax = (corner_idx < 2) ? tmpl.front_axle : tmpl.rear_axle;
    const bool is_right = (corner_idx % 2 == 1);
    const Real axle_x = (corner_idx < 2) ? tmpl.front_axle_x : -tmpl.rear_axle_x;
    const Real track_y = is_right ? -ax.half_track : ax.half_track;

    const Vec3 wheel_center(axle_x, track_y, 0.25);

    auto p = detail::make_mcpherson_params_for_corner(
        wheel_center, ax.mcpherson, is_right, ax.arm_mass, ax.upright_mass);

    auto mc = build_mcpherson_corner(sys, p);
    return {mc, mc.bump_constraint_idx};
}

} // namespace mbd
