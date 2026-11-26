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
#include "mbd/forces/hysteresis.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/drivetrain_params.hpp"
#include "mbd/vehicle/suspension/double_wishbone.hpp"
#include "mbd/vehicle/suspension/mcpherson.hpp"
#include "mbd/vehicle/suspension/multilink.hpp"
#include "mbd/vehicle/suspension/solid_axle.hpp"
#include "mbd/vehicle/suspension/trailing_arm.hpp"
#include "mbd/vehicle/wheel.hpp"

#include <array>
#include <cmath>
#include <limits>
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
    McPherson,      ///< McPherson strut with loop constraints
    FiveLink,       ///< A free upright held by five links (task 7.6)
    TrailingArm,    ///< One arm hinged to the chassis carries the hub (task 7.6); not steered
    SolidAxle       ///< One axle carries both wheels, on four links (task 7.6); not steered
};

/// How a linkage corner's spring unit (its spring, damper and stops) meets
/// the suspension (task 7.6). A McPherson strut's unit is always along the
/// strut, a solid axle's and a trailing arm's always between the chassis
/// and the axle or arm.
enum class SpringLayout {
    CoilOver,   ///< Between the chassis and an arm or the upright
    Pushrod     ///< Double wishbone: a pushrod from the lower arm turns a rocker on the chassis, the unit between rocker and chassis
};

// ============================================================================
// Per-corner hardpoint configuration
// ============================================================================

/// DWB hardpoints expressed relative to the wheel center, for a left corner.
/// These are transformed to world coordinates by the builder using
/// the corner's position on the vehicle; a right corner mirrors Y.
struct DwbHardpoints {
    // Offsets from wheel center (ISO 8855: X forward, Y to the left, which is
    // outboard for a left corner, Z up)
    Vec3 lca_pivot_offset{0.0, -0.40, -0.05};
    Vec3 lca_outer_offset{0.0, -0.08, -0.10};
    Vec3 uca_pivot_offset{0.0, -0.33, 0.13};
    Vec3 uca_outer_offset{0.0, -0.10, 0.10};
    Vec3 tierod_inner_offset{0.10, -0.42, -0.03};
    Vec3 tierod_outer_offset{0.10, -0.08, -0.05};
    Vec3 arm_axis{Vec3::UnitX()};

    // The coil-over (SpringLayout::CoilOver): from the chassis to the lower arm.
    Vec3 spring_chassis_offset{0.0, -0.08, 0.20};
    Vec3 spring_arm_offset{0.0, -0.08, -0.10};

    // The pushrod (SpringLayout::Pushrod): from the lower arm to the rocker,
    // which turns about rocker_axis through rocker_pivot on the chassis; the
    // spring unit from the rocker to the chassis.
    Vec3 pushrod_arm_offset{0.0, -0.12, -0.07};
    Vec3 rocker_pivot_offset{0.0, -0.45, 0.18};
    Vec3 rocker_axis{Vec3::UnitX()};
    Vec3 rocker_pushrod_offset{0.0, -0.40, 0.21};
    Vec3 rocker_spring_offset{0.0, -0.48, 0.25};
    Vec3 pushrod_spring_chassis_offset{0.0, -0.78, 0.25};
};

/// McPherson hardpoints expressed relative to the wheel center, as above.
struct McPhersonHardpoints {
    Vec3 lca_pivot_offset{0.0, -0.40, -0.05};
    Vec3 lca_outer_offset{0.0, -0.10, -0.10};
    Vec3 strut_top_offset{0.0, -0.20, 0.40};
    Vec3 strut_lower_offset{0.0, -0.03, 0.13};
    Vec3 tierod_inner_offset{0.10, -0.42, -0.03};
    Vec3 tierod_outer_offset{0.10, -0.07, -0.05};
    Vec3 arm_axis{Vec3::UnitX()};
};

/// Five-link hardpoints relative to the wheel centre, for a left corner (a
/// right one mirrors Y): each link's inner point (on the chassis; the fifth,
/// the toe link, on the rack of a steered axle) and outer point (on the
/// upright), and the coil-over from the chassis to the upright.
struct FiveLinkHardpoints {
    std::array<Vec3, 5> inner_offset{{Vec3(-0.35, -0.40, -0.13), Vec3(0.00, -0.50, -0.13), Vec3(-0.25, -0.35, 0.14),
                                      Vec3(0.05, -0.45, 0.14), Vec3(0.20, -0.45, -0.06)}};
    std::array<Vec3, 5> outer_offset{{Vec3(-0.05, -0.03, -0.13), Vec3(0.00, -0.03, -0.16), Vec3(-0.05, -0.05, 0.12),
                                      Vec3(0.05, -0.05, 0.12), Vec3(0.10, -0.03, -0.08)}};
    Vec3 spring_chassis_offset{0.0, -0.12, 0.25};
    Vec3 spring_upright_offset{0.0, -0.12, -0.10};
};

/// Trailing-arm hardpoints relative to the wheel centre, for a left corner:
/// the arm's hinge (a point on it and its direction; a right corner mirrors
/// both), and the coil-over from the chassis to the arm.
struct TrailingArmHardpoints {
    Vec3 pivot_offset{0.45, -0.35, 0.02};
    Vec3 axis{Vec3::UnitY()};
    Vec3 spring_chassis_offset{-0.05, -0.15, 0.32};
    Vec3 spring_arm_offset{-0.05, -0.15, 0.0};
};

/// Solid-axle hardpoints relative to the axle's centre (midway between the
/// wheel centres): the lower trailing links (left; the right mirrored), the
/// central upper link, the Panhard rod (its chassis end on the right, its
/// axle end on the left) and the coil-overs (left; mirrored).
struct SolidAxleHardpoints {
    Vec3 lower_chassis_offset{0.50, 0.45, -0.05};
    Vec3 lower_axle_offset{0.0, 0.45, -0.10};
    Vec3 upper_chassis_offset{0.35, 0.0, 0.15};
    Vec3 upper_axle_offset{0.0, 0.0, 0.12};
    Vec3 panhard_chassis_offset{0.0, -0.55, 0.05};
    Vec3 panhard_axle_offset{0.0, 0.55, 0.05};
    Vec3 spring_chassis_offset{0.0, 0.55, 0.30};
    Vec3 spring_axle_offset{0.0, 0.55, 0.0};
};

/// An anti-roll bar acting through drop links (task 7.6): a torsion bar
/// across the car in two halves, each turning in a bearing on the chassis
/// about the bar's lateral axis and pressed by a drop link from its arm's
/// end to the corner (the lower arm of a double wishbone or McPherson, else
/// the upright, the trailing arm or the axle); the bar's torsional
/// stiffness between the halves. Offsets from the left wheel centre; the
/// right side mirrored.
struct AntiRollBarLinks {
    bool enabled{false};   ///< Off: AxleConfig::k_arb acts through VehicleHandle::install_anti_roll_bars
    Vec3 bearing_offset{-0.25, -0.40, 0.05};
    Vec3 arm_end_offset{-0.05, -0.40, 0.05};
    Vec3 link_lower_offset{-0.05, -0.30, -0.08};
    Real arm_mass{1.0};    ///< Each half, a rod from its bearing to its arm's end [kg]
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
    /// A simple corner's spring's free length [m]. A linkage corner's is
    /// found by the builder instead (task 7.6): the length at which its
    /// spring holds the corner at its design position under the car's
    /// weight (detail::design_spring_force).
    Real spring_rest_length{0.35};

    // The spring unit of a linkage corner (task 7.6, D67)
    SpringLayout spring_layout{SpringLayout::CoilOver};
    /// The damper's force against the unit's rate of extension [N against
    /// m/s], in place of c_damper's straight line when not zero: a table
    /// may differ in bump and rebound.
    Curve damper_curve;
    /// The bump stop engages once the unit has shortened by bump_travel from
    /// its length at the design position [m]; then bump_stop gives its force
    /// against its own compression. Infinite: none.
    Real bump_travel{std::numeric_limits<Real>::infinity()};
    Curve bump_stop;
    /// The rebound stop, once the unit has lengthened by rebound_travel.
    Real rebound_travel{std::numeric_limits<Real>::infinity()};
    Curve rebound_stop;
    /// Above zero, the damper acts in series with its top mount of this
    /// stiffness [N/m] (a MaxwellDamper, D66) and the spring alone in the
    /// unit; zero, the damper is in the unit, rigidly mounted.
    Real top_mount_stiffness{0.0};

    // Anti-roll bar
    Real k_arb{0.0};        ///< Roll stiffness [N/m]: the force at each wheel per unit difference of their travels. Zero = no ARB.
    Real c_arb{0.0};        ///< Roll damping [Ns/m].
    /// Through drop links (enabled), the bar's torsional stiffness and
    /// damping are those that give k_arb and c_arb at the wheels at the
    /// design position; otherwise install_anti_roll_bars adds the bar as a
    /// force element.
    AntiRollBarLinks arb_links;

    /// The wheel's mass [kg], a uniform disc of the tyre's free radius
    /// (wheel_inertia), on its spin joint. On a simple corner it is all the
    /// unsprung mass (the slider carrying it has none); a linkage corner adds
    /// its upright and arms.
    Real wheel_mass{40.0};

    /// The axle's tyres (task 7.4): their Magic Formula, and their free
    /// radius, vertical stiffness and damping (UNLOADED_RADIUS,
    /// VERTICAL_STIFFNESS, VERTICAL_DAMPING); Pacejka's Appendix 3 tyre, a
    /// 205/60R15, unless set (mbd/io/tir_file.hpp reads one from a file).
    MagicFormulaParameters tyre{pacejka_appendix3_tyre()};
    /// A tyre model in place of `tyre`'s Magic Formula (task 7.5: a simple
    /// tyre, mbd/forces/simple_tyres.hpp); null, the Magic Formula of `tyre`.
    std::shared_ptr<const TyreModel> tyre_model;

    /// The tyre's radius, vertical stiffness, damping and low speed: the
    /// model's if there is one, else the Magic Formula's.
    TyreBody tyre_body() const { return tyre_model ? tyre_model->body() : MagicFormulaTyre::tyre_body(tyre); }

    /// The tyre's law: `tyre_model`, else the Magic Formula of `tyre`.
    std::shared_ptr<const TyreModel> make_tyre() const
    {
        return tyre_model ? tyre_model : std::make_shared<MagicFormulaTyre>(tyre);
    }

    // Hardpoints (used only when suspension_type != Simple)
    DwbHardpoints dwb;
    McPhersonHardpoints mcpherson;
    FiveLinkHardpoints five_link;
    TrailingArmHardpoints trailing_arm;
    SolidAxleHardpoints solid_axle;

    // Masses of the linkage's parts [kg] (task 7.6, F9: where each lies
    // follows from its joints, mbd/vehicle/suspension/mass.hpp). An arm (a
    // wishbone, a McPherson's lower arm, a trailing arm) is a rod from its
    // pivot to its outer joint; an upright (a trailing arm's hub carrier
    // with it) has its mass among its joints and its wheel centre. A
    // five-link's links have none: their share belongs in the upright's.
    Real arm_mass{5.0};
    Real upright_mass{15.0};
    Real axle_mass{60.0};     ///< A solid axle with its hubs, a rod between its wheel centres
    Real rocker_mass{1.5};    ///< A pushrod's rocker, among its three points
};

// ============================================================================
// Steering configuration
// ============================================================================

/// What steers a steered linkage axle at the front (task 7.7, D68).
enum class SteeringInput {
    /// The rack's travel is prescribed, as an actuator or a robot would: no
    /// column. As before task 7.7.
    RackTravel,
    /// The steering wheel's angle is prescribed; the column, compliant,
    /// turns the pinion, which moves the rack.
    WheelAngle,
    /// A torque on the steering wheel: wheel, column, pinion and rack move as
    /// the forces on them decide.
    WheelTorque
};

struct SteeringConfig {
    /// The rack's stops, with a steering wheel: at the travel that turns a
    /// road wheel at the axle's centre by this angle [rad] (task 7.7).
    Real max_steer_angle{0.6};
    /// Steering wheel angle per road wheel angle at the centre: sets the
    /// pinion's radius from the linkage's rack travel per radian (task 7.7).
    Real steering_ratio{15.0};
    SteeringInput input{SteeringInput::RackTravel};
    Real column_stiffness{2000.0};   ///< Torsional, between the steering wheel and the pinion [N m/rad]
    Real column_damping{2.0};        ///< [N m s/rad]
    /// The steering wheel with the column above its compliance, and the
    /// pinion with the column below it, about the column's axis [kg m^2];
    /// across it 0.6 of that. Their masses are left out (their centres of
    /// mass on the axis).
    Real wheel_inertia{0.04};
    Real pinion_inertia{2e-4};
    /// The column's direction in the chassis frame, across the rack (Y):
    /// from the pinion up and back to the steering wheel.
    Vec3 column_axis{-0.906, 0.0, 0.423};
    Real stop_stiffness{1e6};        ///< The rack's stops [N/m]
};

// ============================================================================
// Chassis configuration
// ============================================================================

struct ChassisConfig {
    Real mass{1400.0};
    Vec3 half_extents{1.5, 0.8, 0.3};
    Vec3 cg_offset{0.0, 0.0, 0.0};  ///< CG offset from geometric center [m]

    // Aerodynamics (optional)
    Real CdA{0.7};                       ///< Drag area [m^2]
    Real ClA{0.0};                       ///< Downforce area [m^2]
    Vec3 aero_cop_offset{Vec3::Zero()};  ///< CoP offset from CG (chassis frame)
    Real aero_h_ref{0.10};
    Real aero_dClA_dh{0.0};
    /// A map of the forces and moments (task 7.10, mbd/forces/aerodynamics.hpp);
    /// when set it replaces the constant model above.
    std::shared_ptr<const AeroMap> aero_map;
    Vec3 aero_reference_point{Vec3::Zero()};   ///< The map's moments are about it (chassis frame)
    Vec3 aero_front_point{Vec3::Zero()};       ///< The front ride height's point (chassis frame)
    Vec3 aero_rear_point{Vec3::Zero()};        ///< The rear's
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

    /// The mass of every body the builder makes [kg]: the chassis, the
    /// wheels, the linkages' arms, uprights, rockers and axles, the steering
    /// racks and the halves of anti-roll bars on drop links (task 7.6, F9).
    Real total_mass() const;

    // --- Presets ---

    static VehicleTemplate DefaultSedan();

    static VehicleTemplate SportsCar();

    static VehicleTemplate FWDHatchback();
};

// ============================================================================
// Steering rack
// ============================================================================

/// A steering command (task 7.7): a value held, or a function of time
/// followed. A held value changes in a step when it is set between steps (the
/// next projection moves the mechanism there); a function's rate and
/// acceleration enter the equations of motion (F10). It is a rack's travel
/// [m], a steering wheel's angle [rad] or the torque on it [N m].
struct SteeringCommand {
    Real held{0.0};
    kernel::TimeFunction path;
    bool follow_path{false};

    Real value(Real t) const { return follow_path ? path.value(t) : held; }
    Real rate(Real t) const { return follow_path ? path.rate(t) : Real(0.0); }
    Real acceleration(Real t) const { return follow_path ? path.acceleration(t) : Real(0.0); }
    /// Hold `x` from now on.
    void hold(Real x)
    {
        held = x;
        follow_path = false;
    }
    /// Follow `f` from now on.
    void follow(kernel::TimeFunction f)
    {
        path = std::move(f);
        follow_path = true;
    }
};

/// A steering rack's command: its travel [m], positive to the left.
using SteeringRack = SteeringCommand;

/// A steering column (task 7.7): the steering wheel and the pinion, each on
/// a hinge about the column's axis on the chassis, a torsion spring between
/// them, the pinion geared to the rack.
struct SteeringColumn {
    BodyIndex wheel_body{0};
    BodyIndex pinion_body{0};
    /// The rack's travel per radian of the pinion [m/rad], signed so that a
    /// positive steering wheel angle turns the car left.
    Real pinion_ratio{0.0};
    Real rack_limit{0.0};                     ///< The rack's stops [m]
    RotationalSpringDamper* column{nullptr};  ///< The compliance
    /// The steering wheel's angle [rad] (WheelAngle) or the torque on it [N m]
    /// (WheelTorque).
    std::shared_ptr<SteeringCommand> command;
};

// ============================================================================
// Per-corner data in the vehicle handle
// ============================================================================

struct CornerHandle {
    /// The body that carries the wheel: the upright of a linkage corner, the
    /// slider of a simple one. It steers (linkage corners), takes the spring
    /// and the anti-roll bar, and the brake reacts on it. Its origin is the
    /// wheel centre; toe and camber are read from it.
    BodyIndex upright_body{0};
    BodyIndex wheel_body{0};       ///< The wheel, on its spin joint on the upright; the tyre's body
    BodyIndex lca_body{0};         ///< LCA body (0 for Simple suspension)
    BodyIndex uca_body{0};         ///< UCA body (0 for Simple/McPherson)
    FullTireForce* tire{nullptr};
    SuspensionType type{SuspensionType::Simple};

    // Steering of a linkage corner (DWB, McPherson) on a steered axle: the tie
    // rod's inner point is on the axle's rack. Simple corners steer the tyre.
    bool on_rack{false};                   ///< Tie rod attached to a steering rack
    Real rack_per_rad{0.0};                ///< Rack travel per radian of toe, calibrated
    std::size_t tierod_constraint_idx{0};  ///< Index in System::constraints (linkage corners)

    // The spring unit of a linkage corner (task 7.6, D67)
    /// The wheel centre in the upright's frame: zero but for a solid axle,
    /// whose frame is at its centre.
    Vec3 wheel_center_in_upright{Vec3::Zero()};
    BodyIndex rocker_body{0};     ///< The pushrod's rocker (SpringLayout::Pushrod), 0 otherwise
    /// Where the anti-roll bar's drop link meets the corner: the lower arm of
    /// a double wishbone or McPherson, else the upright (trailing arm, axle).
    BodyIndex arb_link_body{0};
    /// That body's frame's origin at the reference, in the chassis's frame
    /// (its axes are the chassis's there).
    Vec3 arb_link_origin{Vec3::Zero()};
    BodyIndex arb_half_body{0};   ///< This side's half of an anti-roll bar on drop links, 0 otherwise
    /// The unit's ends, each a point in its body's frame: the chassis's (or
    /// the rocker's frame's carrier) first, the corner's second.
    kernel::Marker spring_end1, spring_end2;
    SpringDamper* spring{nullptr};       ///< The unit's spring and, without a top mount, its damper
    SpringDamper* stops{nullptr};        ///< Its bump and rebound stops, when it has any
    MaxwellDamper* top_mount{nullptr};   ///< Its damper in series with its top mount, when it has one
    Real design_spring_force{0.0};       ///< The spring's force at the design position, pushing apart [N]
    Real design_spring_length{0.0};      ///< The unit's length there [m]
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

    /// Each axle's tyre load at rest, per wheel [N], from the masses the
    /// builder made and where they lie (task 7.6): what the linkage
    /// corners' springs were set to carry at their design position.
    std::array<Real, 2> design_tyre_load{{0.0, 0.0}};
    /// The torsion of each axle's anti-roll bar on drop links (front, rear;
    /// null without one): its stiffness and damping are those that give the
    /// axle's k_arb and c_arb at the wheels.
    std::array<RotationalSpringDamper*, 2> arb_torsion{{nullptr, nullptr}};

    /// The front axle's steering column, when the template's steering input
    /// is the steering wheel's angle or a torque on it (task 7.7); empty
    /// (bodies 0) otherwise.
    SteeringColumn column;

    // Convenience accessors
    FullTireForce* tire(int c) { return corners[c].tire; }
    const FullTireForce* tire(int c) const { return corners[c].tire; }
    BodyIndex wheel(int c) const { return corners[c].wheel_body; }
    BodyIndex upright(int c) const { return corners[c].upright_body; }

    /// Steer by delta, the angle of a single front wheel at the axle's centre
    /// (bicycle model). Simple corners get their Ackermann angle directly as
    /// tyre steer angle. Linkage axles move their rack by delta times the
    /// axle's mean calibrated ratio; the toe of each wheel then follows from
    /// the linkage geometry. Takes effect at the next step's projection.
    ///
    /// With a steering wheel (task 7.7): its angle is held at steering_ratio
    /// delta (WheelAngle), a step that the column passes on to the pinion and
    /// the rack dynamically; a steered rear axle's rack as before. A car
    /// steered by a torque (WheelTorque) throws MBD-V005: use
    /// set_steering_torque or steer.
    void set_steering(Real delta);

    /// Steer the front axle by a function of time (task 7.7): its rack's
    /// travel [m] (RackTravel), its steering wheel's angle [rad] (WheelAngle)
    /// or the torque on it [N m] (WheelTorque). Its rate and acceleration
    /// enter the equations of motion. Throws MBD-V005 for a car whose front
    /// axle is not steered by a rack.
    void steer(const kernel::TimeFunction& f);

    /// Hold a torque on the steering wheel [N m], positive turning it left
    /// (WheelTorque; MBD-V005 otherwise).
    void set_steering_torque(Real torque);

    void clear_steering()
    {
        set_steering(0.0);
    }

    /// Put every tyre on `road` (mbd/forces/road.hpp); each starts on a level
    /// road at height 0.
    void set_road(const std::shared_ptr<const Road>& road);

    /// Install anti-roll bars on front and/or rear axles based on template config.
    /// Must be called AFTER the vehicle is built and positioned at equilibrium,
    /// but before simulation. Returns pointers to the installed ARB force elements
    /// (front first, then rear) for runtime parameter adjustment. Nullptr if no
    /// ARB is installed on that axle: none asked, or one on drop links, which
    /// build_vehicle made (task 7.6).
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

/// Mirror a hardpoint for the right side of the vehicle: ISO 8855 has +Y to
/// the left, so right side hardpoints have Y negated.
inline Vec3 mirror_y(const Vec3& v) { return Vec3(v.x(), -v.y(), v.z()); }

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

/// Convert McPherson offset hardpoints to world coordinates for a given corner.
McPhersonParams make_mcpherson_params_for_corner(
    const Vec3& wheel_center_chassis,
    const McPhersonHardpoints& hp,
    bool is_right_side,
    Real arm_mass,
    Real upright_mass);

/// The same for a five-link corner, a trailing-arm corner, and a solid axle
/// (whose centre is at (axle_x, 0, 0)).
MultilinkParams make_five_link_params_for_corner(const Vec3& wheel_center_chassis, const AxleConfig& ax,
                                                 bool is_right_side);
TrailingArmParams make_trailing_arm_params_for_corner(const Vec3& wheel_center_chassis, const AxleConfig& ax,
                                                      bool is_right_side);
SolidAxleParams make_solid_axle_params(Real axle_x, const AxleConfig& ax);

/// The mechanism of one linkage corner (double wishbone, McPherson,
/// five-link, trailing arm) on `chassis_body`, as build_vehicle makes it
/// (task 7.6): its bodies, joints and constraints, its wheel, a pushrod's
/// rocker, and where its spring unit's ends and its drop link are; not its
/// spring unit nor its tyre. The tie rod's (toe link's) inner point is on
/// `tierod_body`. Throws MBD-V002 for a steered trailing arm.
CornerHandle build_linkage_corner(kernel::System& sys, BodyIndex chassis_body, const Vec3& wheel_center_chassis,
                                  const AxleConfig& ax, bool is_right_side, BodyIndex tierod_body,
                                  const std::string& name);

/// A solid axle's mechanism, its two wheels and its spring units' ends (left
/// corner first), its centre at (axle_x, 0, 0); not its spring units nor its
/// tyres. Throws MBD-V002 for a steered solid axle.
std::array<CornerHandle, 2> build_solid_axle_corners(kernel::System& sys, BodyIndex chassis_body, Real axle_x,
                                                     const AxleConfig& ax, const std::string& name);

/// One half of an anti-roll bar on drop links, on the side of `corner` (its
/// axle's left wheel centre at left_wheel_center; the right side mirrored):
/// the half on a hinge about Y at its bearing on the chassis, its frame
/// there with the chassis's axes at the reference, and its drop link to the
/// corner. The torsion between the halves is added by build_vehicle, once
/// the bar's motion is known.
void build_anti_roll_bar_half(kernel::System& sys, BodyIndex chassis_body, const Vec3& left_wheel_center,
                              const AxleConfig& ax, CornerHandle& corner, bool is_right_side);

/// The mass of a steering rack [kg] (add_steering_rack).
inline constexpr Real steering_rack_mass = 1.0;

/// True when the axle's anti-roll bar acts through drop links: asked for,
/// on a linkage (a simple corner has nothing for a drop link to meet).
inline bool anti_roll_bar_on_links(const AxleConfig& ax)
{
    return ax.arb_links.enabled && ax.suspension_type != SuspensionType::Simple;
}

/// How a linkage corner's parts move with its wheel at the design position,
/// per unit of wheel-centre travel up (task 7.6): one side, or a solid axle's
/// two in heave.
struct CornerMotion {
    int sides{1};
    std::array<Real, 2> spring_length{{0.0, 0.0}};   ///< Each spring unit's length [m]
    std::array<Real, 2> spring_rate{{0.0, 0.0}};     ///< Its rate of extension [m/m]: the motion ratio
    std::array<Real, 2> contact_rate{{0.0, 0.0}};    ///< The rate of rise of each tyre's contact point [m/m]
    /// Each upright's turn about its tyre's heading [rad/m]: the camber's
    /// rate, along which the tyre's overturning moment does work.
    std::array<Real, 2> camber_rate{{0.0, 0.0}};
    std::array<Real, 2> bar_rate{{0.0, 0.0}};        ///< Each anti-roll bar half's turn about Y [rad/m]
    /// sum of m_i dz_i/dt over the moving bodies: the rate at which their
    /// weight does work, over g [kg].
    Real weight_rate{0.0};
};

/// The motion of axle `axle`'s left corner (0 front, 1 rear; or of a solid
/// axle) of `tmpl` at its design position, the tyres at `loaded_radius`, from
/// the velocities of build_corner_rig's mechanism (Kinematics). Throws
/// MBD-V003 when the mechanism does not solve or its spring unit does not
/// move with the wheel.
CornerMotion corner_motion(const VehicleTemplate& tmpl, int axle, Real loaded_radius);

/// The force (pushing apart, [N]) each spring unit must carry for the
/// corner to rest at its design position with each tyre carrying
/// tyre_load, by virtual work along the motion: the springs' power, the
/// tyres' and the weights' add to zero,
///     F sum(spring_rate) + tyre_load sum(contact_rate)
///       + overturning (camber_rate[0] - camber_rate[1]) - g weight_rate = 0,
/// overturning the left tyre's moment about its heading at rest (the right
/// one's is its mirror image).
Real design_spring_force(const CornerMotion& m, Real tyre_load, Real overturning);

/// Add the corner's spring unit between its ends: the spring (free length
/// so that it pushes with design_force at design_length), the damper (or a
/// MaxwellDamper with the top mount), and the stops, each measured from
/// design_length. Fills the corner's spring, stops, top_mount and design
/// values.
void add_spring_unit(kernel::System& sys, const AxleConfig& ax, CornerHandle& corner, Real design_force,
                     Real design_length);

/// A steering rack: a light bar sliding along the chassis Y (lateral) axis,
/// positive travel to the left, its frame the chassis frame at zero travel,
/// driven by its command when `rack` is not null.
BodyIndex add_steering_rack(kernel::System& sys, BodyIndex chassis_body,
                                   const std::shared_ptr<SteeringRack>& rack,
                                   const std::string& name);

/// A steering column on `chassis_body` for the rack `rack_body` (task 7.7):
/// the pinion and the steering wheel, each on a hinge about the column's
/// axis through the chassis frame's origin; the pinion geared to the rack by
/// a JointCoupler, rack travel = pinion_ratio times the pinion's angle; the
/// column's torsion spring between them; the rack's stops at +-rack_limit; and
/// the input: a driver on the steering wheel's hinge (WheelAngle) or a torque
/// on it (WheelTorque), each following the column's command. Throws MBD-V005
/// for RackTravel, a column axis along the rack, or a ratio of zero.
SteeringColumn add_steering_column(kernel::System& sys, BodyIndex chassis_body, BodyIndex rack_body,
                                   const SteeringConfig& config, Real pinion_ratio, Real rack_limit);

/// Rack travel per radian of toe for one corner: the corner alone on a fixed
/// chassis, its rack moved by 5 mm, the toe measured before and after.
/// Returns 0 if the mechanism cannot be solved or is not on a rack.
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
///   - 4 corners with suspension, springs, dampers, and tires, each wheel on
///     its spin joint on the corner's upright (task 7.3)
///
/// The vehicle layer uses ISO 8855 axes (X forward, Y left, Z up; plan task
/// 7.1, mbd/vehicle/axes.hpp): the model's gravity is set to -Z. The tyres
/// start on a level road at height 0; VehicleHandle::set_road gives them
/// another.
///
/// Returns a VehicleHandle with convenient accessors.
VehicleHandle build_vehicle(kernel::System& sys,
                                   const VehicleTemplate& tmpl = VehicleTemplate::DefaultSedan());

// ============================================================================
// Equilibrium solver for the template-built vehicle
// ============================================================================

/// Set every wheel of the vehicle spinning at its free-rolling rate for the
/// simulator's present state (set_free_rolling): after giving the chassis a
/// speed, so that the tyres do not start by skidding.
void roll_wheels(kernel::Simulator& sim, const VehicleHandle& vh);

/// Place a template-built vehicle at rest at its static equilibrium: the
/// simulator's state is set and its kinematics refreshed.
///
/// A car with a linkage corner (task 7.6): each linkage corner at its design
/// position, which its spring was set to hold under design_tyre_load, each
/// simple corner at its spring's length under its load less its wheel's
/// weight, the chassis raised and pitched so that every wheel centre is at
/// its tyre's loaded radius above the road. At rest there to within the
/// gravity's small turn into the pitched chassis's axes.
///
/// A car of simple corners only keeps the placement it had before: its
/// chassis at the mean of the heights the axles' springs give under a share
/// of the whole weight, unpitched (near equilibrium; the sedan's regression
/// scenarios start there).
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

/// A linkage corner of a template alone on a fixed chassis, for analysis
/// (task 7.6): what build_corner_rig made.
struct CornerRig {
    BodyIndex chassis_body{1};                ///< On a fixed joint at the world's origin
    std::vector<CornerHandle> corners;        ///< One, or a solid axle's two (left first)
    std::vector<std::size_t> drivers;         ///< The bump drivers, one per corner, in sys.constraints
};

/// The corner `corner_idx` (FL = 0, FR, RL, RR) of `tmpl` as build_vehicle
/// makes its mechanism, on a fixed chassis, its tie rod on the chassis, with
/// a bump driver per wheel holding the wheel centre's height at its design
/// value (zero: the chassis frame's origin height) plus t; no spring unit,
/// tyre or anti-roll bar torsion, but an anti-roll bar's half and drop link
/// if the axle has them. A solid axle comes whole, both drivers +t (heave).
/// Throws MBD-V002 for a simple corner.
CornerRig build_corner_rig(kernel::System& sys, const VehicleTemplate& tmpl, int corner_idx);

} // namespace mbd
