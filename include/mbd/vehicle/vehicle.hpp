#pragma once

// Simplified full vehicle model builder.
//
// Creates a 5-body vehicle on the kinematics kernel: chassis on a free joint
// + 4 wheels on prismatic joints for vertical suspension travel (10 DOF).
//
// q layout: [tx, ty, tz, quaternion (x, y, z, w), q_FL, q_FR, q_RL, q_RR]
// v layout: [chassis w, chassis v (both in chassis axes), qd_FL ... qd_RR]
// The suspension coordinate is the travel (positive = wheel moves down from
// its mount). Use VehicleModel::susp_q_idx and susp_v_idx rather than fixed
// offsets.

#include "mbd/kernel/simulator.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/tire.hpp"

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
        1.5, 0.3, 0.8};             ///< [m] (half-length, half-height, half-width)

    // --- Geometry ---
    Real front_axle_x{1.35};         ///< Distance from CG to front axle [m]
    Real rear_axle_x{1.35};          ///< Distance from CG to rear axle [m]
    Real half_track{0.8};            ///< Half track width [m]

    // --- Wheels ---
    Real wheel_mass{40.0};           ///< Per-wheel unsprung mass [kg]
    Vec3 wheel_half_extents{         ///< For inertia computation
        0.15, 0.15, 0.15};          ///< [m]

    // --- Suspension ---
    Real k_susp{25000.0};            ///< Spring stiffness per corner [N/m]
    Real c_susp{2000.0};             ///< Damping per corner [Ns/m]
    Real susp_rest_length{0.35};     ///< Spring free length [m]

    // --- Tires ---
    Real tire_free_radius{0.35};     ///< Unloaded radius [m]
    Real tire_k_z{200000.0};         ///< Vertical stiffness [N/m]
    Real tire_c_z{500.0};            ///< Vertical damping [Ns/m]
    PacejkaTireParams tire_params{PacejkaTireParams::DefaultPassengerCar()};

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
        return weight_per_wheel() / tire_k_z;
    }

    Real wheel_center_height_eq() const
    {
        return tire_free_radius - tire_deflection_eq();
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
    std::array<BodyIndex, 4> wheel_bodies{2, 3, 4, 5};
    std::array<int, 4> wheel_joint_indices{};
    int chassis_joint_index{0};
    std::array<int, 4> susp_q{};   ///< Index of each suspension coordinate in q
    std::array<int, 4> susp_v{};   ///< Index of each suspension velocity in v
    std::array<FullTireForce*, 4> tires{};
    VehicleParams params;

    /// Index of the chassis height (ty) in q.
    int chassis_ty_idx() const { return 1; }

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
};

// ============================================================================
// Builder function
// ============================================================================

/// Build a simplified vehicle on a kernel system. Returns a VehicleModel
/// with indices for easy access. The vehicle layer is still Y-up with +Z to
/// the left (ISO 8855 is plan task 7.1), so the model's gravity is set to -Y.
VehicleModel build_simple_vehicle(kernel::System& sys,
                                         const VehicleParams& p = VehicleParams{});

/// Put the simulator's state at the static equilibrium of the vehicle, at rest.
void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleModel& vm);

} // namespace mbd
