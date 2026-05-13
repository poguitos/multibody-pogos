#pragma once

// Vehicle template: hierarchical configuration + unified builder.
//
// Usage:
//   VehicleTemplate tmpl = VehicleTemplate::DefaultSedan();
//   MultibodySystem sys;
//   auto vh = build_vehicle(sys, tmpl);
//   Simulator sim(sys);
//   ...

#include "mbd/system.hpp"
#include "mbd/tire.hpp"
#include "mbd/constraint.hpp"
#include "mbd/drivetrain_params.hpp"
#include "mbd/double_wishbone.hpp"
#include "mbd/mcpherson.hpp"
#include "mbd/anti_roll_bar.hpp"
#include "mbd/aerodynamics.hpp"

#include <array>
#include <string>
#include <utility>
#include <cmath>

namespace mbd {

// ============================================================================
// Suspension type enum
// ============================================================================

enum class SuspensionType {
    Simple,         ///< Single prismatic joint (simplest, fastest)
    DoubleWishbone, ///< DWB with loop constraints (kinematically accurate)
    McPherson       ///< McPherson strut with loop constraints
};

// ============================================================================
// Per-corner hardpoint configuration
// ============================================================================

/// DWB hardpoints expressed relative to the wheel center.
/// These are transformed to world coordinates by the builder using
/// the corner's position on the vehicle.
struct DwbHardpoints {
    // Offsets from wheel center (positive X = forward, Y = up, Z = outboard)
    Vec3 lca_pivot_offset{0.0, -0.05, -0.40};
    Vec3 lca_outer_offset{0.0, -0.10, -0.08};
    Vec3 uca_pivot_offset{0.0, 0.13, -0.33};
    Vec3 uca_outer_offset{0.0, 0.10, -0.10};
    Vec3 tierod_inner_offset{0.10, -0.03, -0.42};
    Vec3 tierod_outer_offset{0.10, -0.05, -0.08};
    Vec3 arm_axis{Vec3::UnitX()};
};

/// McPherson hardpoints expressed relative to the wheel center.
struct McPhersonHardpoints {
    Vec3 lca_pivot_offset{0.0, -0.05, -0.40};
    Vec3 lca_outer_offset{0.0, -0.10, -0.10};
    Vec3 strut_top_offset{0.0, 0.40, -0.20};
    Vec3 strut_lower_offset{0.0, 0.13, -0.03};
    Vec3 tierod_inner_offset{0.10, -0.03, -0.42};
    Vec3 tierod_outer_offset{0.10, -0.05, -0.07};
    Vec3 arm_axis{Vec3::UnitX()};
};

// ============================================================================
// Axle configuration
// ============================================================================

struct AxleConfig {
    SuspensionType suspension_type{SuspensionType::Simple};
    bool is_steered{false};

    // Geometry
    Real half_track{0.8};

    // Spring-damper
    Real k_spring{25000.0};
    Real c_damper{2000.0};
    Real spring_rest_length{0.35};

    // Anti-roll bar
    Real k_arb{0.0};        ///< Roll stiffness [N/m]. Zero = no ARB.
    Real c_arb{0.0};        ///< Roll damping [Ns/m].

    // Wheel/unsprung mass
    Real wheel_mass{40.0};
    Vec3 wheel_half_extents{0.15, 0.15, 0.15};

    // Tire
    Real tire_free_radius{0.35};
    Real tire_k_z{200000.0};
    Real tire_c_z{500.0};
    PacejkaTireParams tire_params{PacejkaTireParams::DefaultPassengerCar()};

    // Hardpoints (used only when suspension_type != Simple)
    DwbHardpoints dwb;
    McPhersonHardpoints mcpherson;

    // Suspension arm mass (for DWB/McPherson bodies)
    // These must be large enough that the mass matrix is well-conditioned.
    // Real arms are typically 3-8 kg each.
    Real arm_mass{5.0};
    Real upright_mass{15.0};
};

// ============================================================================
// Steering configuration
// ============================================================================

struct SteeringConfig {
    Real max_steer_angle{0.6};  ///< Maximum driver input [rad] (~34 deg)
    Real steering_ratio{15.0};  ///< Steering wheel turns to road wheel angle
};

// ============================================================================
// Chassis configuration
// ============================================================================

struct ChassisConfig {
    Real mass{1400.0};
    Vec3 half_extents{1.5, 0.3, 0.8};
    Vec3 cg_offset{0.0, 0.0, 0.0};  ///< CG offset from geometric center [m]

    // Aerodynamics (optional)
    Real CdA{0.7};                       ///< Drag area [m^2]
    Real ClA{0.0};                       ///< Downforce area [m^2]
    Vec3 aero_cop_offset{Vec3::Zero()};  ///< CoP offset from CG (chassis frame)
    Real aero_h_ref{0.10};
    Real aero_dClA_dh{0.0};
};

// ============================================================================
// Complete vehicle template
// ============================================================================

struct VehicleTemplate {
    std::string name{"default_vehicle"};

    ChassisConfig chassis;
    AxleConfig front_axle;
    AxleConfig rear_axle;
    SteeringConfig steering;
    DrivetrainParams drivetrain;

    // Axle positions relative to chassis CG
    Real front_axle_x{1.35};
    Real rear_axle_x{1.35};

    Real wheelbase() const { return front_axle_x + rear_axle_x; }

    Real total_mass() const
    {
        return chassis.mass
             + 2.0 * front_axle.wheel_mass
             + 2.0 * rear_axle.wheel_mass;
    }

    // --- Presets ---

    static VehicleTemplate DefaultSedan()
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

    static VehicleTemplate SportsCar()
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

    static VehicleTemplate FWDHatchback()
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
};

// ============================================================================
// Per-corner data in the vehicle handle
// ============================================================================

struct CornerHandle {
    BodyIndex wheel_body{0};       ///< Wheel/upright body (tire attaches here)
    BodyIndex lca_body{0};         ///< LCA body (0 for Simple suspension)
    BodyIndex uca_body{0};         ///< UCA body (0 for Simple/McPherson)
    FullTireForce* tire{nullptr};
    SuspensionType type{SuspensionType::Simple};

    // Steering: for DWB/McPherson, we steer by moving the tie rod inner point.
    // These are nullptr/zero for Simple suspension (which uses tire->steer_angle).
    DistanceConstraint* tierod_constraint{nullptr};
    Vec3 tierod_inner_ref{Vec3::Zero()};  ///< Reference tie rod inner point (chassis frame)
    Real rack_per_rad{0.0};               ///< Calibration: rack-Z-motion per rad of wheel toe
};

// ============================================================================
// Vehicle handle (result of build_vehicle)
// ============================================================================

struct VehicleHandle {
    VehicleTemplate tmpl;

    BodyIndex chassis_body{1};
    std::array<CornerHandle, 4> corners; ///< FL=0, FR=1, RL=2, RR=3

    // Convenience accessors
    FullTireForce* tire(int c) { return corners[c].tire; }
    const FullTireForce* tire(int c) const { return corners[c].tire; }
    BodyIndex wheel(int c) const { return corners[c].wheel_body; }

    /// Apply Ackermann steering to steered axles.
    /// For Simple suspension: sets tire->steer_angle (kinematic).
    /// For DWB/McPherson: moves the tie rod inner point (produces real toe via geometry).
    void set_steering(Real delta)
    {
        // Compute a single rack displacement from the average of front
        // calibration ratios. The rack is physically one rigid bar — both
        // tie rod inner points move together by the same vector.
        Real rack_per_rad_avg = 0.0;
        int n_steered = 0;
        if (tmpl.front_axle.is_steered) {
            for (int i = 0; i < 2; ++i) {
                if (corners[i].rack_per_rad != 0.0) {
                    rack_per_rad_avg += std::abs(corners[i].rack_per_rad);
                    ++n_steered;
                }
            }
        }
        if (n_steered > 0) rack_per_rad_avg /= n_steered;

        auto apply_to_corner = [this, &rack_per_rad_avg](int idx, Real target_angle) {
            auto& c = corners[idx];
            if (c.type == SuspensionType::Simple) {
                if (c.tire) c.tire->steer_angle = target_angle;
            } else {
                // DWB / McPherson: use tie rod motion
                if (c.tire) c.tire->steer_angle = 0.0;  // Zero out kinematic steer
                if (c.tierod_constraint != nullptr && rack_per_rad_avg != 0.0) {
                    // Use uniform rack displacement based on FL convention.
                    // For positive target_angle (left turn), rack shifts in +Z direction.
                    const Real rack_disp = target_angle * rack_per_rad_avg;
                    c.tierod_constraint->anchor1_B =
                        c.tierod_inner_ref + Vec3(0.0, 0.0, rack_disp);
                }
            }
        };

        if (std::abs(delta) < Real(1e-10)) {
            for (int idx = 0; idx < 4; ++idx) {
                apply_to_corner(idx, 0.0);
            }
            return;
        }

        const Real L = tmpl.wheelbase();
        const Real R = L / std::tan(delta);

        if (tmpl.front_axle.is_steered) {
            const Real ht = tmpl.front_axle.half_track;
            apply_to_corner(0, std::atan(L / (R - ht)));
            apply_to_corner(1, std::atan(L / (R + ht)));
        } else {
            apply_to_corner(0, 0.0);
            apply_to_corner(1, 0.0);
        }

        if (tmpl.rear_axle.is_steered) {
            const Real ht = tmpl.rear_axle.half_track;
            apply_to_corner(2, std::atan(L / (R - ht)));
            apply_to_corner(3, std::atan(L / (R + ht)));
        } else {
            apply_to_corner(2, 0.0);
            apply_to_corner(3, 0.0);
        }
    }

    void clear_steering()
    {
        set_steering(0.0);
    }

    /// Install anti-roll bars on front and/or rear axles based on template config.
    /// Must be called AFTER the vehicle is built and positioned at equilibrium,
    /// but before simulation. Returns pointers to the installed ARB force elements
    /// (front first, then rear) for runtime parameter adjustment. Nullptr if no
    /// ARB is installed on that axle.
    std::pair<AntiRollBar*, AntiRollBar*> install_anti_roll_bars(
        MultibodySystem& sys,
        const std::vector<RigidBodyState>& equilibrium_states)
    {
        AntiRollBar* front_arb = nullptr;
        AntiRollBar* rear_arb  = nullptr;

        if (tmpl.front_axle.k_arb > 0.0) {
            auto arb = std::make_unique<AntiRollBar>(
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
            auto arb = std::make_unique<AntiRollBar>(
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

    /// Install aerodynamic forces on the chassis based on chassis config.
    /// Returns a pointer for runtime parameter adjustment, or nullptr if
    /// no aero is configured (CdA = 0 and ClA = 0).
    AerodynamicForce* install_aerodynamics(MultibodySystem& sys)
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

        auto aero = std::make_unique<AerodynamicForce>(chassis_body, p);
        AerodynamicForce* ptr = aero.get();
        sys.force_elements.push_back(std::move(aero));
        return ptr;
    }
};

// ============================================================================
// Builder: internal helpers
// ============================================================================

namespace detail {

/// Mirror hardpoint Z coordinate for the right side of the vehicle.
/// Our convention: +Z = left. Right side hardpoints have Z negated.
inline Vec3 mirror_z(const Vec3& v) { return Vec3(v.x(), v.y(), -v.z()); }

/// Build a simple (prismatic) corner.
inline CornerHandle build_simple_corner(
    MultibodySystem& sys,
    BodyIndex chassis_body,
    const Vec3& mount_pos_chassis,
    const AxleConfig& axle,
    const std::string& name)
{
    CornerHandle ch;
    ch.type = SuspensionType::Simple;

    // Prismatic joint: chassis → wheel, axis = chassis -Y (downward)
    const Mat3 R_susp = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();

    auto I_wheel = RigidBodyInertia::from_solid_box(axle.wheel_mass, axle.wheel_half_extents);
    ch.wheel_body = sys.add_body(I_wheel, RigidBodyState{}, name, chassis_body);

    Transform3 X_PJ(R_susp, mount_pos_chassis);
    Transform3 X_CJ = Transform3::FromRotation(R_susp);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_PJ, X_CJ, chassis_body, ch.wheel_body));

    // Spring-damper
    sys.force_elements.push_back(std::make_unique<LinearSpringDamper>(
        chassis_body, ch.wheel_body,
        mount_pos_chassis, Vec3::Zero(),
        axle.k_spring, axle.c_damper, axle.spring_rest_length));

    // Tire
    auto tire = std::make_unique<FullTireForce>(
        ch.wheel_body, axle.tire_free_radius,
        axle.tire_k_z, axle.tire_c_z, axle.tire_params);
    ch.tire = tire.get();
    sys.force_elements.push_back(std::move(tire));

    return ch;
}

/// Convert DWB offset hardpoints to world coordinates for a given corner.
inline DoubleWishboneParams make_dwb_params_for_corner(
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

/// Build a DWB corner for dynamic simulation.
inline CornerHandle build_dwb_corner(
    MultibodySystem& sys,
    BodyIndex chassis_body,
    const Vec3& wheel_center_chassis,
    const AxleConfig& axle,
    bool is_right_side,
    const std::string& /*name*/)
{
    CornerHandle ch;
    ch.type = SuspensionType::DoubleWishbone;

    // Construct DWB hardpoints for this corner (in chassis frame)
    auto p = make_dwb_params_for_corner(
        wheel_center_chassis, axle.dwb, is_right_side,
        axle.arm_mass, axle.upright_mass);

    // Build the dynamic DWB mechanism parented to the chassis
    auto dwb = build_double_wishbone_corner_dynamic(sys, chassis_body, p);

    ch.lca_body     = dwb.lca_body;
    ch.uca_body     = dwb.uca_body;
    ch.wheel_body   = dwb.upright_body;

    // Track the tie rod constraint for steering.
    // The tie rod is the LAST constraint added by build_double_wishbone_corner_dynamic.
    ch.tierod_constraint = dynamic_cast<DistanceConstraint*>(
        sys.constraints[dwb.tierod_constraint_idx].get());
    ch.tierod_inner_ref = p.tierod_inner;

    // Spring-damper: from chassis mount (above LCA outer) to LCA outer point.
    // This models a coil-over-arm spring layout.
    const Vec3 spring_chassis_mount = p.lca_outer + Vec3(0.0, 0.30, 0.0);
    const Vec3 spring_lca_attach_chassis = p.lca_outer;
    // Convert chassis-frame spring attachment points to LCA body frame.
    // LCA body frame has identity orientation at reference; origin at lca_pivot.
    const Vec3 spring_lca_attach_body = spring_lca_attach_chassis - p.lca_pivot;

    // Spring rest length: reference geometric distance + static precompression.
    // Precompression is chosen so that the spring supports ~1/4 of vehicle
    // weight at reference. Since the caller doesn't pass the vehicle weight
    // here, we use a representative value of ~4000 N per corner.
    const Real ref_distance = (spring_chassis_mount - (p.lca_pivot + spring_lca_attach_body)).norm();
    const Real representative_corner_load = 4000.0;  // N, approximate quarter-weight
    const Real precompression = representative_corner_load / axle.k_spring;
    const Real spring_rest_dyn = ref_distance + precompression;

    sys.force_elements.push_back(std::make_unique<LinearSpringDamper>(
        chassis_body, dwb.lca_body,
        spring_chassis_mount,
        spring_lca_attach_body,
        axle.k_spring, axle.c_damper, spring_rest_dyn));
        
    // Tire: attaches to upright (wheel center is the upright origin)
    auto tire = std::make_unique<FullTireForce>(
        dwb.upright_body,
        axle.tire_free_radius,
        axle.tire_k_z,
        axle.tire_c_z,
        axle.tire_params);
    ch.tire = tire.get();
    sys.force_elements.push_back(std::move(tire));

    return ch;
}
/// Convert McPherson offset hardpoints to world coordinates for a given corner.
inline McPhersonParams make_mcpherson_params_for_corner(
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

/// Build a McPherson corner for dynamic simulation.
inline CornerHandle build_mcpherson_corner(
    MultibodySystem& sys,
    BodyIndex chassis_body,
    const Vec3& wheel_center_chassis,
    const AxleConfig& axle,
    bool is_right_side,
    const std::string& /*name*/)
{
    CornerHandle ch;
    ch.type = SuspensionType::McPherson;

    // Construct McPherson hardpoints for this corner (in chassis frame)
    auto p = make_mcpherson_params_for_corner(
        wheel_center_chassis, axle.mcpherson, is_right_side,
        axle.arm_mass, axle.upright_mass);

    // Build the dynamic McPherson mechanism parented to the chassis
    auto mc = mbd::build_mcpherson_corner_dynamic(sys, chassis_body, p);

    ch.lca_body   = mc.lca_body;
    ch.uca_body   = 0; // no UCA in McPherson
    ch.wheel_body = mc.upright_body;

    // Track the tie rod constraint for steering.
    ch.tierod_constraint = dynamic_cast<DistanceConstraint*>(
        sys.constraints[mc.tierod_constraint_idx].get());
    ch.tierod_inner_ref = p.tierod_inner;

    // Spring-damper: from strut top mount (chassis side) to strut lower attachment on upright.
    // The strut spring acts along the strut axis between these two points.
    // Convert strut_lower (chassis frame) to upright body frame
    const Vec3 spring_upright_attach = p.strut_lower - p.wheel_center;

    const Real ref_distance = (p.strut_top_mount - (p.wheel_center + spring_upright_attach)).norm();
    const Real representative_corner_load = 4000.0;
    const Real precompression = representative_corner_load / axle.k_spring;
    const Real spring_rest_dyn = ref_distance + precompression;

    sys.force_elements.push_back(std::make_unique<LinearSpringDamper>(
        chassis_body, mc.upright_body,
        p.strut_top_mount,
        spring_upright_attach,
        axle.k_spring, axle.c_damper, spring_rest_dyn));

    // Tire attaches to upright
    auto tire = std::make_unique<FullTireForce>(
        mc.upright_body,
        axle.tire_free_radius,
        axle.tire_k_z,
        axle.tire_c_z,
        axle.tire_params);
    ch.tire = tire.get();
    sys.force_elements.push_back(std::move(tire));

    return ch;
}

} // namespace detail

// ============================================================================
// Main builder function
// ============================================================================

/// Build a complete vehicle MultibodySystem from a template.
///
/// The builder creates:
///   - Chassis on FreeCoordJoint (6 DOF)
///   - 4 corners with suspension, springs, dampers, and tires
///
/// q layout for simple suspension:
///   [tx, ty, tz, rx, ry, rz, q_FL, q_FR, q_RL, q_RR]
///
/// Returns a VehicleHandle with convenient accessors.
inline VehicleHandle build_vehicle(MultibodySystem& sys,
                                   const VehicleTemplate& tmpl = VehicleTemplate::DefaultSedan())
{
    VehicleHandle vh;
    vh.tmpl = tmpl;

    // --- Chassis (body 1) ---
    auto I_chassis = RigidBodyInertia::from_solid_box(
        tmpl.chassis.mass, tmpl.chassis.half_extents);
    vh.chassis_body = sys.add_body(
        I_chassis, RigidBodyState{}, tmpl.name + "_chassis", kGroundIndex);

    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, vh.chassis_body));

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
        const auto& cd = corner_defs[c];

        switch (cd.axle.suspension_type) {
            case SuspensionType::Simple:
                vh.corners[c] = detail::build_simple_corner(
                    sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.name);
                break;
            case SuspensionType::DoubleWishbone:
                vh.corners[c] = detail::build_dwb_corner(
                    sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.is_right, cd.name);
                break;
            case SuspensionType::McPherson:
                vh.corners[c] = detail::build_mcpherson_corner(
                    sys, vh.chassis_body, cd.mount_pos, cd.axle, cd.is_right, cd.name);
                break;
        }
    }

    // --- Calibrate rack-to-wheel-angle ratio for each steered DWB/McPherson corner ---
    // Strategy: for each corner with a tie rod constraint, apply a small rack
    // displacement, solve constraints (with chassis held fixed at identity),
    // measure the resulting toe angle, compute ratio = rack_motion / toe_angle.
    //
    // We can't easily "hold chassis fixed" in an already-built free-joint vehicle,
    // so instead we use a local subsystem approach: build a test system with
    // the same corner on a fixed chassis, calibrate, then discard.
    for (int c = 0; c < 4; ++c) {
        auto& corner = vh.corners[c];
        if (corner.tierod_constraint == nullptr) continue; // Simple suspension or non-steered

        const auto& ax = (c < 2) ? tmpl.front_axle : tmpl.rear_axle;
        const bool is_right = (c % 2 == 1);

        // Build a calibration subsystem: fixed chassis + this corner
        MultibodySystem sys_calib;
        auto I_chassis_calib = RigidBodyInertia::from_solid_box(
            10000.0, tmpl.chassis.half_extents);
        BodyIndex chassis_calib = sys_calib.add_body(
            I_chassis_calib, RigidBodyState{}, "calib_chassis", kGroundIndex);
        sys_calib.add_joint(std::make_unique<FixedJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, chassis_calib));

        const Vec3 wheel_center_chassis = corner_defs[c].mount_pos;

        BodyIndex upright_calib = 0;
        DistanceConstraint* calib_tierod = nullptr;
        Vec3 calib_tierod_ref = Vec3::Zero();

        if (ax.suspension_type == SuspensionType::DoubleWishbone) {
            auto p = detail::make_dwb_params_for_corner(
                wheel_center_chassis, ax.dwb, is_right,
                ax.arm_mass, ax.upright_mass);
            auto dwb_calib = build_double_wishbone_corner_dynamic(
                sys_calib, chassis_calib, p);
            upright_calib = dwb_calib.upright_body;
            calib_tierod = dynamic_cast<DistanceConstraint*>(
                sys_calib.constraints[dwb_calib.tierod_constraint_idx].get());
            calib_tierod_ref = p.tierod_inner;
        } else if (ax.suspension_type == SuspensionType::McPherson) {
            auto p = detail::make_mcpherson_params_for_corner(
                wheel_center_chassis, ax.mcpherson, is_right,
                ax.arm_mass, ax.upright_mass);
            auto mc_calib = mbd::build_mcpherson_corner_dynamic(
                sys_calib, chassis_calib, p);
            upright_calib = mc_calib.upright_body;
            calib_tierod = dynamic_cast<DistanceConstraint*>(
                sys_calib.constraints[mc_calib.tierod_constraint_idx].get());
            calib_tierod_ref = p.tierod_inner;
        } else {
            continue;
        }

        if (calib_tierod == nullptr) continue;

        // Reference configuration
        sys_calib.q.setZero();
        sys_calib.compute_kinematics();
        bool ok = solve_position_kinematics(sys_calib, 50, 1e-10);
        if (!ok) continue;
        Real toe_ref = extract_toe(sys_calib.states[upright_calib]);

        // Perturb rack by +5mm in chassis +Z direction
        const Real rack_step = 0.005;
        calib_tierod->anchor1_B = calib_tierod_ref + Vec3(0.0, 0.0, rack_step);
        ok = solve_position_kinematics(sys_calib, 50, 1e-10);
        if (!ok) {
            calib_tierod->anchor1_B = calib_tierod_ref;
            continue;
        }
        Real toe_perturbed = extract_toe(sys_calib.states[upright_calib]);
        const Real dtoe = toe_perturbed - toe_ref;

        // Ratio: rack displacement per radian of toe
        if (std::abs(dtoe) > 1e-6) {
            corner.rack_per_rad = rack_step / dtoe;
        }

        // Restore (calib subsystem is local, about to go out of scope anyway)
        calib_tierod->anchor1_B = calib_tierod_ref;
    }

    return vh;
}

// ============================================================================
// Equilibrium solver for the template-built vehicle
// ============================================================================

/// Compute approximate static equilibrium for a template-built vehicle.
///
/// For Simple corners: chassis_y = wheel_y + spring_equilibrium_length.
/// For DWB/McPherson corners: chassis_y = wheel_y - wheel_center_in_chassis.y(),
/// because the DWB mechanism at reference places the wheel at
/// wheel_center_chassis relative to the chassis origin.
inline void set_vehicle_equilibrium(MultibodySystem& sys,
                                    const VehicleHandle& vh)
{
    const auto& t = vh.tmpl;

    sys.q.setZero();
    sys.q_dot.setZero();

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

    // Chassis height depends on suspension type.
    // For Simple: chassis_y = wheel_y + suspension_length (spring compressed).
    // For DWB/McPherson at reference: chassis_y = wheel_y - wheel_center_chassis.y().
    // We pick the FIRST front corner's type to determine chassis height.
    Real chassis_y_front = 0.0;
    switch (vh.corners[0].type) {
        case SuspensionType::Simple: {
            const Real spring_compr = W_front_per / t.front_axle.k_spring;
            const Real susp_length = t.front_axle.spring_rest_length - spring_compr;
            chassis_y_front = wheel_y_front_world + susp_length;
            break;
        }
        case SuspensionType::DoubleWishbone:
        case SuspensionType::McPherson: {
            // Wheel center in chassis frame = at reference position
            // We assume a wheel_center_y_chassis of 0.25 (standard hardpoint set).
            // Actually we need the per-corner mount_pos Y used by the builder,
            // which for DWB/McPherson is the wheel_center position at corner.
            // For our template, the wheel center in chassis frame is at:
            //   y = 0 (the mount_pos y is 0 in build_vehicle's corner_defs)
            // Wait — the DWB builder uses wheel_center_chassis which is passed
            // as the corner mount_pos. In build_vehicle, mount_pos has y=0.
            // But the hardpoint offsets position the wheel above/below that.
            // Actually the DWB builder treats wheel_center_chassis as the wheel
            // center itself. Let's re-check.
            //
            // In build_vehicle the corner_def mount_pos = (x, 0, z). This is
            // the WHEEL CENTER in chassis frame. So wheel center Y in chassis
            // = 0. At reference, chassis at identity → wheel Y in world = 0.
            // For tire contact at ground: chassis_y = tire_free_radius.
            chassis_y_front = t.front_axle.tire_free_radius
                            - W_front_per / t.front_axle.tire_k_z;
            break;
        }
    }

    Real chassis_y_rear = 0.0;
    switch (vh.corners[2].type) {
        case SuspensionType::Simple: {
            const Real spring_compr = W_rear_per / t.rear_axle.k_spring;
            const Real susp_length = t.rear_axle.spring_rest_length - spring_compr;
            chassis_y_rear = wheel_y_rear_world + susp_length;
            break;
        }
        case SuspensionType::DoubleWishbone:
        case SuspensionType::McPherson: {
            chassis_y_rear = t.rear_axle.tire_free_radius
                           - W_rear_per / t.rear_axle.tire_k_z;
            break;
        }
    }

    sys.q(1) = 0.5 * (chassis_y_front + chassis_y_rear);

    // Set per-corner DOFs
    int q_idx = 6;
    for (int c = 0; c < 4; ++c) {
        const auto& ax = (c < 2) ? t.front_axle : t.rear_axle;
        const Real wheel_y = (c < 2) ? wheel_y_front_world : wheel_y_rear_world;

        switch (vh.corners[c].type) {
            case SuspensionType::Simple:
                // q = chassis_y - wheel_y (suspension travel)
                sys.q(q_idx) = sys.q(1) - wheel_y;
                q_idx += 1;
                break;
            case SuspensionType::DoubleWishbone:
                // 5 tree DOFs at zero (reference config)
                q_idx += 5;
                break;
            case SuspensionType::McPherson:
                // 4 tree DOFs at zero
                q_idx += 4;
                break;
        }
    }

    sys.compute_kinematics();
}

// ============================================================================
// Kinematic analysis helper: build standalone corner for sweep
// ============================================================================

/// Build a standalone DWB corner (ground-mounted) for kinematic analysis.
/// Returns the corner handle and the bump constraint index.
inline std::pair<DoubleWishboneCorner, size_t> build_dwb_for_analysis(
    MultibodySystem& sys,
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

/// Build a standalone McPherson corner for kinematic analysis.
inline std::pair<McPhersonCorner, size_t> build_mcpherson_for_analysis(
    MultibodySystem& sys,
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