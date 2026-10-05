#pragma once

// Vehicle template: hierarchical configuration + unified builder, on the
// kinematics kernel (plan task 2.7).
//
// Usage:
//   VehicleTemplate tmpl = VehicleTemplate::DefaultSedan();
//   kernel::System sys;
//   auto vh = build_vehicle(sys, tmpl);
//   kernel::Simulator sim(sys);
//   set_vehicle_equilibrium(sim, vh);
//   sim.initialize();
//   ...

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/forces/aerodynamics.hpp"
#include "mbd/forces/anti_roll_bar.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/drivetrain_params.hpp"
#include "mbd/vehicle/suspension/double_wishbone.hpp"
#include "mbd/vehicle/suspension/mcpherson.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

    static VehicleTemplate DefaultSedan();

    static VehicleTemplate SportsCar();

    static VehicleTemplate FWDHatchback();
};

// ============================================================================
// Steering rack
// ============================================================================

/// Commanded travel of a steering rack [m], read by the rack's driver. Set it
/// through VehicleHandle::set_steering between steps: the driver's target
/// changes, and the projection after the next step moves the mechanism there.
struct SteeringRack {
    Real travel{0.0};
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

    // Steering of a linkage corner (DWB, McPherson) on a steered axle: the tie
    // rod's inner point is on the axle's rack. Simple corners steer the tyre.
    bool on_rack{false};                   ///< Tie rod attached to a steering rack
    Real rack_per_rad{0.0};                ///< Rack travel per radian of toe, calibrated
    std::size_t tierod_constraint_idx{0};  ///< Index in System::constraints (linkage corners)
};

// ============================================================================
// Vehicle handle (result of build_vehicle)
// ============================================================================

struct VehicleHandle {
    VehicleTemplate tmpl;

    BodyIndex chassis_body{1};
    std::array<CornerHandle, 4> corners; ///< FL=0, FR=1, RL=2, RR=3

    /// Steering racks of the front and rear axles: present when the axle is
    /// steered and has linkage suspension (rack body index 0 otherwise).
    std::array<BodyIndex, 2> rack_body{{0, 0}};
    std::array<std::shared_ptr<SteeringRack>, 2> rack;

    // Convenience accessors
    FullTireForce* tire(int c) { return corners[c].tire; }
    const FullTireForce* tire(int c) const { return corners[c].tire; }
    BodyIndex wheel(int c) const { return corners[c].wheel_body; }

    /// Steer by delta, the angle of a single front wheel at the axle's centre
    /// (bicycle model). Simple corners get their Ackermann angle directly as
    /// tyre steer angle. Linkage axles move their rack by delta times the
    /// axle's mean calibrated ratio; the toe of each wheel then follows from
    /// the linkage geometry. Takes effect at the next step's projection.
    void set_steering(Real delta);

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
        kernel::System& sys,
        const std::vector<RigidBodyState>& equilibrium_states);

    /// Install aerodynamic forces on the chassis based on chassis config.
    /// Returns a pointer for runtime parameter adjustment, or nullptr if
    /// no aero is configured (CdA = 0 and ClA = 0).
    AerodynamicForce* install_aerodynamics(kernel::System& sys);
};

// ============================================================================
// Builder: internal helpers
// ============================================================================

namespace detail {

/// Mirror hardpoint Z coordinate for the right side of the vehicle.
/// Our convention: +Z = left. Right side hardpoints have Z negated.
inline Vec3 mirror_z(const Vec3& v) { return Vec3(v.x(), v.y(), -v.z()); }

/// Build a simple (prismatic) corner.
CornerHandle build_simple_corner(
    kernel::System& sys,
    BodyIndex chassis_body,
    const Vec3& mount_pos_chassis,
    const AxleConfig& axle,
    const std::string& name);

/// Convert DWB offset hardpoints to world coordinates for a given corner.
DoubleWishboneParams make_dwb_params_for_corner(
    const Vec3& wheel_center_chassis,
    const DwbHardpoints& hp,
    bool is_right_side,
    Real arm_mass,
    Real upright_mass);

/// Spring rest length for a linkage corner: the reference distance between
/// its attachments plus a precompression that carries about a quarter of the
/// vehicle's weight (the caller does not pass that weight here; 4000 N is the
/// representative value used throughout).
Real precompressed_rest_length(Real reference_distance, Real k_spring);

/// Build a DWB corner for dynamic simulation. The tie rod's inner point is on
/// `tierod_body` (the chassis, or the axle's steering rack).
CornerHandle build_dwb_corner(
    kernel::System& sys,
    BodyIndex chassis_body,
    const Vec3& wheel_center_chassis,
    const AxleConfig& axle,
    bool is_right_side,
    BodyIndex tierod_body);

/// Convert McPherson offset hardpoints to world coordinates for a given corner.
McPhersonParams make_mcpherson_params_for_corner(
    const Vec3& wheel_center_chassis,
    const McPhersonHardpoints& hp,
    bool is_right_side,
    Real arm_mass,
    Real upright_mass);

/// Build a McPherson corner for dynamic simulation. The tie rod's inner point
/// is on `tierod_body` (the chassis, or the axle's steering rack).
CornerHandle build_mcpherson_corner(
    kernel::System& sys,
    BodyIndex chassis_body,
    const Vec3& wheel_center_chassis,
    const AxleConfig& axle,
    bool is_right_side,
    BodyIndex tierod_body);

/// A steering rack: a light bar sliding along the chassis Z (lateral) axis,
/// its frame the chassis frame at zero travel, driven to rack->travel.
BodyIndex add_steering_rack(kernel::System& sys, BodyIndex chassis_body,
                                   const std::shared_ptr<SteeringRack>& rack,
                                   const std::string& name);

/// Rack travel per radian of toe for one corner: the corner alone on a fixed
/// chassis, its rack moved by 5 mm, the toe measured before and after.
/// Returns 0 if the mechanism cannot be solved.
Real calibrate_rack(const VehicleTemplate& tmpl, const AxleConfig& ax,
                           const Vec3& wheel_center_chassis, bool is_right);

} // namespace detail

// ============================================================================
// Main builder function
// ============================================================================

/// Build a complete vehicle on a kernel system from a template.
///
/// The builder creates:
///   - Chassis on a free joint (body 1)
///   - A steering rack for each steered axle with linkage suspension
///   - 4 corners with suspension, springs, dampers, and tires
///
/// The vehicle layer is still Y-up with +Z to the left (ISO 8855 is plan
/// task 7.1), so the model's gravity is set to -Y.
///
/// Returns a VehicleHandle with convenient accessors.
VehicleHandle build_vehicle(kernel::System& sys,
                                   const VehicleTemplate& tmpl = VehicleTemplate::DefaultSedan());

// ============================================================================
// Equilibrium solver for the template-built vehicle
// ============================================================================

/// Place a template-built vehicle near its static equilibrium, at rest: the
/// simulator's state is set and its kinematics refreshed.
///
/// For Simple corners: chassis_y = wheel_y + spring_equilibrium_length.
/// For DWB/McPherson corners: the mechanism at its reference puts the wheel
/// centre at the chassis origin's height, so chassis_y is the loaded wheel
/// centre height.
void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleHandle& vh);

// ============================================================================
// Kinematic analysis helper: build standalone corner for sweep
// ============================================================================

/// Build a standalone DWB corner (ground-mounted) for kinematic analysis,
/// with its bump driver (see Kinematics). Returns the corner handle and the
/// index of the bump driver.
std::pair<DoubleWishboneCorner, std::size_t> build_dwb_for_analysis(
    kernel::System& sys,
    const VehicleTemplate& tmpl,
    int corner_idx);

/// Build a standalone McPherson corner for kinematic analysis.
std::pair<McPhersonCorner, std::size_t> build_mcpherson_for_analysis(
    kernel::System& sys,
    const VehicleTemplate& tmpl,
    int corner_idx);

} // namespace mbd
