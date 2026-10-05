#include "mbd/vehicle/vehicle_template.hpp"

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
    t.chassis.half_extents = Vec3(1.4, 0.25, 0.75);

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
    t.chassis.half_extents = Vec3(1.2, 0.28, 0.72);

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

std::pair<AntiRollBar*, AntiRollBar*> VehicleHandle::install_anti_roll_bars(
    kernel::System& sys,
    const std::vector<RigidBodyState>& equilibrium_states)
{
    AntiRollBar* front_arb = nullptr;
    AntiRollBar* rear_arb  = nullptr;

    if (tmpl.front_axle.k_arb > 0.0) {
        auto arb = std::make_shared<AntiRollBar>(
            chassis_body,
            corners[0].wheel_body,  // FL
            corners[1].wheel_body,  // FR
            tmpl.front_axle.k_arb,
            tmpl.front_axle.c_arb);
        arb->capture_reference(equilibrium_states);
        front_arb = arb.get();
        sys.force_elements.push_back(std::move(arb));
    }

    if (tmpl.rear_axle.k_arb > 0.0) {
        auto arb = std::make_shared<AntiRollBar>(
            chassis_body,
            corners[2].wheel_body,  // RL
            corners[3].wheel_body,  // RR
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

    // Prismatic joint: chassis → wheel, axis = chassis -Y (downward)
    const Mat3 R_susp = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    ch.wheel_body = sys.model.add_body(
        chassis_body, std::make_shared<kernel::PrismaticJointModel>(),
        Transform3(R_susp, mount_pos_chassis), Transform3::FromRotation(R_susp),
        RigidBodyInertia::from_solid_box(axle.wheel_mass, axle.wheel_half_extents), name);

    // Spring-damper
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        chassis_body, ch.wheel_body,
        mount_pos_chassis, Vec3::Zero(),
        axle.k_spring, axle.c_damper, axle.spring_rest_length));

    // Tire
    auto tire = std::make_shared<FullTireForce>(
        ch.wheel_body, axle.tire_free_radius,
        axle.tire_k_z, axle.tire_c_z, axle.tire_params);
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
        Vec3 world = wheel_center_chassis + (is_right_side ? mirror_z(v) : v);
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

Real detail::precompressed_rest_length(Real reference_distance, Real k_spring)
{
    const Real representative_corner_load = 4000.0;  // N
    return reference_distance + representative_corner_load / k_spring;
}

CornerHandle detail::build_dwb_corner(
    kernel::System& sys,
    BodyIndex chassis_body,
    const Vec3& wheel_center_chassis,
    const AxleConfig& axle,
    bool is_right_side,
    BodyIndex tierod_body)
{
    CornerHandle ch;
    ch.type = SuspensionType::DoubleWishbone;

    const auto p = make_dwb_params_for_corner(
        wheel_center_chassis, axle.dwb, is_right_side,
        axle.arm_mass, axle.upright_mass);
    const auto dwb = build_double_wishbone_corner_dynamic(sys, chassis_body, p, tierod_body);

    ch.lca_body     = dwb.lca_body;
    ch.uca_body     = dwb.uca_body;
    ch.wheel_body   = dwb.upright_body;
    ch.on_rack      = tierod_body != chassis_body;
    ch.tierod_constraint_idx = dwb.tierod_constraint_idx;

    // Spring-damper: from chassis mount (above LCA outer) to LCA outer point.
    // This models a coil-over-arm spring layout. The LCA body frame has the
    // chassis axes at the reference, origin at lca_pivot.
    const Vec3 spring_chassis_mount = p.lca_outer + Vec3(0.0, 0.30, 0.0);
    const Vec3 spring_lca_attach_body = p.lca_outer - p.lca_pivot;
    const Real ref_distance = (spring_chassis_mount - p.lca_outer).norm();

    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        chassis_body, dwb.lca_body,
        spring_chassis_mount,
        spring_lca_attach_body,
        axle.k_spring, axle.c_damper,
        precompressed_rest_length(ref_distance, axle.k_spring)));

    // Tire: attaches to upright (wheel center is the upright origin)
    auto tire = std::make_shared<FullTireForce>(
        dwb.upright_body,
        axle.tire_free_radius,
        axle.tire_k_z,
        axle.tire_c_z,
        axle.tire_params);
    ch.tire = tire.get();
    sys.force_elements.push_back(std::move(tire));

    return ch;
}

McPhersonParams detail::make_mcpherson_params_for_corner(
    const Vec3& wheel_center_chassis,
    const McPhersonHardpoints& hp,
    bool is_right_side,
    Real arm_mass,
    Real upright_mass)
{
    auto offset = [&](const Vec3& v) -> Vec3 {
        Vec3 world = wheel_center_chassis + (is_right_side ? mirror_z(v) : v);
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

CornerHandle detail::build_mcpherson_corner(
    kernel::System& sys,
    BodyIndex chassis_body,
    const Vec3& wheel_center_chassis,
    const AxleConfig& axle,
    bool is_right_side,
    BodyIndex tierod_body)
{
    CornerHandle ch;
    ch.type = SuspensionType::McPherson;

    const auto p = make_mcpherson_params_for_corner(
        wheel_center_chassis, axle.mcpherson, is_right_side,
        axle.arm_mass, axle.upright_mass);
    const auto mc = mbd::build_mcpherson_corner_dynamic(sys, chassis_body, p, tierod_body);

    ch.lca_body   = mc.lca_body;
    ch.uca_body   = 0; // no UCA in McPherson
    ch.wheel_body = mc.upright_body;
    ch.on_rack    = tierod_body != chassis_body;
    ch.tierod_constraint_idx = mc.tierod_constraint_idx;

    // Spring-damper along the strut: from the top mount (chassis) to the
    // strut's lower attachment on the upright.
    const Vec3 spring_upright_attach = p.strut_lower - p.wheel_center;
    const Real ref_distance = (p.strut_top_mount - p.strut_lower).norm();

    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        chassis_body, mc.upright_body,
        p.strut_top_mount,
        spring_upright_attach,
        axle.k_spring, axle.c_damper,
        precompressed_rest_length(ref_distance, axle.k_spring)));

    // Tire attaches to upright
    auto tire = std::make_shared<FullTireForce>(
        mc.upright_body,
        axle.tire_free_radius,
        axle.tire_k_z,
        axle.tire_c_z,
        axle.tire_params);
    ch.tire = tire.get();
    sys.force_elements.push_back(std::move(tire));

    return ch;
}

BodyIndex detail::add_steering_rack(kernel::System& sys, BodyIndex chassis_body,
                                    const std::shared_ptr<SteeringRack>& rack,
                                    const std::string& name)
{
    const BodyIndex body = sys.model.add_body(
        chassis_body, std::make_shared<kernel::PrismaticJointModel>(),
        Transform3::Identity(), Transform3::Identity(),
        RigidBodyInertia::from_solid_box(1.0, Vec3(0.02, 0.02, 0.4)), name);
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
    kernel::System calib;
    const BodyIndex chassis = calib.model.add_body(
        0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
        Transform3::Identity(),
        RigidBodyInertia::from_solid_box(10000.0, tmpl.chassis.half_extents), "calib_chassis");
    auto rack = std::make_shared<SteeringRack>();
    const BodyIndex rack_body = add_steering_rack(calib, chassis, rack, "calib_rack");

    BodyIndex upright = 0;
    if (ax.suspension_type == SuspensionType::DoubleWishbone) {
        const auto p = make_dwb_params_for_corner(wheel_center_chassis, ax.dwb, is_right,
                                                  ax.arm_mass, ax.upright_mass);
        upright = build_double_wishbone_corner_dynamic(calib, chassis, p, rack_body).upright_body;
    } else if (ax.suspension_type == SuspensionType::McPherson) {
        const auto p = make_mcpherson_params_for_corner(wheel_center_chassis, ax.mcpherson,
                                                        is_right, ax.arm_mass, ax.upright_mass);
        upright = build_mcpherson_corner_dynamic(calib, chassis, p, rack_body).upright_body;
    } else {
        return 0.0;
    }

    Kinematics k(calib);
    if (!k.solve(50, 1e-10)) return 0.0;
    const Real toe_ref = extract_toe(k.state(upright));

    const Real rack_step = 0.005;   // +5 mm toward +Z (left)
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
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);

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
        {Vec3( tmpl.front_axle_x, 0.0,  tmpl.front_axle.half_track), tmpl.front_axle, false, "FL"},
        {Vec3( tmpl.front_axle_x, 0.0, -tmpl.front_axle.half_track), tmpl.front_axle, true,  "FR"},
        {Vec3(-tmpl.rear_axle_x,  0.0,  tmpl.rear_axle.half_track),  tmpl.rear_axle,  false, "RL"},
        {Vec3(-tmpl.rear_axle_x,  0.0, -tmpl.rear_axle.half_track),  tmpl.rear_axle,  true,  "RR"},
    }};

    for (int c = 0; c < 4; ++c) {
        const auto& cd = corner_defs[static_cast<std::size_t>(c)];
        const BodyIndex rack_body = vh.rack_body[c / 2];
        const BodyIndex tierod_body = rack_body > 0 ? rack_body : vh.chassis_body;

        switch (cd.axle.suspension_type) {
            case SuspensionType::Simple:
                vh.corners[c] = detail::build_simple_corner(
                    sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.name);
                break;
            case SuspensionType::DoubleWishbone:
                vh.corners[c] = detail::build_dwb_corner(
                    sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.is_right, tierod_body);
                break;
            case SuspensionType::McPherson:
                vh.corners[c] = detail::build_mcpherson_corner(
                    sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.is_right, tierod_body);
                break;
        }
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

void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleHandle& vh)
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

    // Per-corner world-frame wheel center height (contact at y=0)
    const Real wheel_y_front_world =
        t.front_axle.tire_free_radius - W_front_per / t.front_axle.tire_k_z;
    const Real wheel_y_rear_world =
        t.rear_axle.tire_free_radius - W_rear_per / t.rear_axle.tire_k_z;

    auto chassis_height = [](const CornerHandle& corner, const AxleConfig& ax, Real wheel_y, Real W_per) {
        if (corner.type == SuspensionType::Simple) {
            const Real spring_compr = W_per / ax.k_spring;
            return wheel_y + (ax.spring_rest_length - spring_compr);
        }
        return wheel_y;
    };
    const Real chassis_y_front = chassis_height(vh.corners[0], t.front_axle, wheel_y_front_world, W_front_per);
    const Real chassis_y_rear  = chassis_height(vh.corners[2], t.rear_axle, wheel_y_rear_world, W_rear_per);
    const Real chassis_y = 0.5 * (chassis_y_front + chassis_y_rear);

    // The free joint's coordinates start with the translation.
    sim.q(model.idx_q[vh.chassis_body] + 1) = chassis_y;

    // Simple corners: the prismatic coordinate is the suspension travel.
    for (int c = 0; c < 4; ++c) {
        if (vh.corners[c].type != SuspensionType::Simple) continue;
        const Real wheel_y = (c < 2) ? wheel_y_front_world : wheel_y_rear_world;
        sim.q(model.idx_q[vh.corners[c].wheel_body]) = chassis_y - wheel_y;
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
    const Real track_z = is_right ? -ax.half_track : ax.half_track;

    const Vec3 wheel_center(axle_x, 0.25, track_z);

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
    const Real track_z = is_right ? -ax.half_track : ax.half_track;

    const Vec3 wheel_center(axle_x, 0.25, track_z);

    auto p = detail::make_mcpherson_params_for_corner(
        wheel_center, ax.mcpherson, is_right, ax.arm_mass, ax.upright_mass);

    auto mc = build_mcpherson_corner(sys, p);
    return {mc, mc.bump_constraint_idx};
}

} // namespace mbd
