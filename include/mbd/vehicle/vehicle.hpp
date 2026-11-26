#pragma once

// Simplified full vehicle model builder.
//
// Creates a 9-body vehicle on the kinematics kernel: chassis on a free joint;
// at each corner a carrier on a prismatic joint for vertical suspension
// travel, and the wheel on its spin joint on the carrier (task 7.3,
// mbd/vehicle/wheel.hpp): 14 DOF.
//
// q layout: [tx, ty, tz, quaternion (x, y, z, w), then for each corner FL,
//            FR, RL, RR its travel and its wheel's spin angle]
// v layout: [chassis w, chassis v (both in chassis axes), then for each
//            corner its travel rate and its wheel's spin rate]
// The suspension coordinate is the travel (positive = wheel moves down from
// its mount). Use VehicleModel::susp_q_idx, susp_v_idx and spin_v rather than
// fixed offsets.

#include "mbd/kernel/simulator.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/vehicle/wheel.hpp"

#include <array>
#include <utility>

namespace mbd {

// ============================================================================
// Vehicle parameters
// ============================================================================

struct VehicleParams {
    // --- Chassis ---
    Real chassis_mass{1400.0};       ///< [kg]
    Vec3 chassis_half_extents{       ///< For inertia computation
        1.5, 0.8, 0.3};             ///< [m] (half-length, half-width, half-height)

    // --- Geometry ---
    Real front_axle_x{1.35};         ///< Distance from CG to front axle [m]
    Real rear_axle_x{1.35};          ///< Distance from CG to rear axle [m]
    Real half_track{0.8};            ///< Half track width [m]

    // --- Wheels ---
    /// Per-corner unsprung mass [kg], all of it in the wheel, a uniform disc
    /// of the tyre's free radius (wheel_inertia); the carrier has none.
    Real wheel_mass{40.0};

    // --- Suspension ---
    Real k_susp{25000.0};            ///< Spring stiffness per corner [N/m]
    Real c_susp{2000.0};             ///< Damping per corner [Ns/m]
    Real susp_rest_length{0.35};     ///< Spring free length [m]

    // --- Tyres ---
    /// Every corner's tyre (task 7.4): its Magic Formula, and its free radius,
    /// vertical stiffness and damping (UNLOADED_RADIUS, VERTICAL_STIFFNESS,
    /// VERTICAL_DAMPING); Pacejka's Appendix 3 tyre, a 205/60R15, unless set
    /// (mbd/io/tir_file.hpp reads one from a file).
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

    // --- Derived quantities ---

    Real total_mass() const
    {
        return chassis_mass + 4.0 * wheel_mass;
    }

    Real weight_per_wheel() const
    {
        return total_mass() * g_accel / 4.0;
    }

    Real tire_deflection_eq() const
    {
        return weight_per_wheel() / tyre_body().vertical_stiffness;
    }

    Real wheel_center_height_eq() const
    {
        return tyre_body().free_radius - tire_deflection_eq();
    }

    Real spring_force_eq() const
    {
        return chassis_mass * g_accel / 4.0;
    }

    Real spring_compression_eq() const
    {
        return spring_force_eq() / k_susp;
    }

    Real susp_length_eq() const
    {
        return susp_rest_length - spring_compression_eq();
    }

    /// Static equilibrium suspension travel (positive = wheel below mount).
    /// This equals the equilibrium spring length since at q=0 the spring has
    /// zero length, and the spring stretches as q increases.
    Real q_susp_eq() const
    {
        return susp_length_eq();
    }

    /// Chassis CG height at static equilibrium.
    Real chassis_height_eq() const
    {
        return wheel_center_height_eq() + q_susp_eq();
    }
};

// ============================================================================
// Corner identifiers
// ============================================================================

enum class Corner { FL = 0, FR = 1, RL = 2, RR = 3 };

// ============================================================================
// Vehicle model handle (provides convenient access to indices)
// ============================================================================

struct VehicleModel {
    BodyIndex chassis_body{1};
    /// Each corner's carrier: the body on the suspension's slider, which the
    /// spring pulls and the brake reacts on. Its origin is the wheel centre.
    std::array<BodyIndex, 4> carrier_bodies{2, 4, 6, 8};
    /// Each corner's wheel, on its spin joint on the carrier: the tyre's body.
    std::array<BodyIndex, 4> wheel_bodies{3, 5, 7, 9};
    std::array<int, 4> wheel_joint_indices{};   ///< The carriers' (suspension) joints
    int chassis_joint_index{0};
    std::array<int, 4> susp_q{};   ///< Index of each suspension coordinate in q
    std::array<int, 4> susp_v{};   ///< Index of each suspension velocity in v
    std::array<int, 4> spin_v{};   ///< Index of each wheel's spin rate (relative to its carrier) in v
    std::array<FullTireForce*, 4> tires{};
    VehicleParams params;

    /// Index of the chassis height (tz, ISO 8855's up) in q.
    int chassis_tz_idx() const { return 2; }

    /// Index of a wheel's suspension coordinate in q.
    int susp_q_idx(Corner c) const { return susp_q[static_cast<std::size_t>(c)]; }

    /// Index of a wheel's suspension velocity in v.
    int susp_v_idx(Corner c) const { return susp_v[static_cast<std::size_t>(c)]; }
    /// Compute Ackermann steering angles for front wheels.
    /// \param delta  Driver steering input [rad]. Positive = left turn.
    /// \return {delta_FL, delta_FR}
    std::pair<Real, Real> ackermann_steering(Real delta) const;

    /// Apply Ackermann steering to front tires.
    /// \param delta  Driver steering input [rad]. Positive = left turn.
    void set_front_steering(Real delta);

    /// Set all four tire steering angles individually.
    void set_steering_angles(Real fl, Real fr, Real rl, Real rr);

    /// Clear all steering (set all angles to zero).
    void clear_steering();

    /// Put every tyre on `road` (mbd/forces/road.hpp); each starts on a level
    /// road at height 0.
    void set_road(const std::shared_ptr<const Road>& road);
};

/// Set every wheel of the vehicle spinning at its free-rolling rate for the
/// simulator's present state (set_free_rolling): after giving the chassis a
/// speed, so that the tyres do not start by skidding.
void roll_wheels(kernel::Simulator& sim, const VehicleModel& vm);

// ============================================================================
// Builder function
// ============================================================================

/// Build a simplified vehicle on a kernel system. Returns a VehicleModel
/// with indices for easy access. The vehicle layer uses ISO 8855 axes (X
/// forward, Y left, Z up; mbd/vehicle/axes.hpp): the model's gravity is set
/// to -Z. The tyres start on a level road at height 0; set_road gives them
/// another.
VehicleModel build_simple_vehicle(kernel::System& sys,
                                         const VehicleParams& p = VehicleParams{});

/// Put the simulator's state at the static equilibrium of the vehicle, at rest.
void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleModel& vm);

} // namespace mbd
