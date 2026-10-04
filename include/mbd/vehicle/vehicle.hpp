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
    std::pair<Real, Real> ackermann_steering(Real delta) const
    {
        if (std::abs(delta) < Real(1e-10)) return {Real(0.0), Real(0.0)};

        const Real L  = params.front_axle_x + params.rear_axle_x;
        const Real ht = params.half_track;
        const Real R  = L / std::tan(delta);

        const Real delta_FL = std::atan(L / (R - ht));
        const Real delta_FR = std::atan(L / (R + ht));

        return {delta_FL, delta_FR};
    }

    /// Apply Ackermann steering to front tires.
    /// \param delta  Driver steering input [rad]. Positive = left turn.
    void set_front_steering(Real delta)
    {
        auto [d_FL, d_FR] = ackermann_steering(delta);
        tires[0]->steer_angle = d_FL;
        tires[1]->steer_angle = d_FR;
    }

    /// Set all four tire steering angles individually.
    void set_steering_angles(Real fl, Real fr, Real rl, Real rr)
    {
        tires[0]->steer_angle = fl;
        tires[1]->steer_angle = fr;
        tires[2]->steer_angle = rl;
        tires[3]->steer_angle = rr;
    }

    /// Clear all steering (set all angles to zero).
    void clear_steering()
    {
        for (auto* t : tires) {
            t->steer_angle = Real(0.0);
        }
    }
};

// ============================================================================
// Builder function
// ============================================================================

/// Build a simplified vehicle on a kernel system. Returns a VehicleModel
/// with indices for easy access. The vehicle layer is still Y-up with +Z to
/// the left (ISO 8855 is plan task 7.1), so the model's gravity is set to -Y.
inline VehicleModel build_simple_vehicle(kernel::System& sys,
                                         const VehicleParams& p = VehicleParams{})
{
    VehicleModel vm;
    vm.params = p;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);

    // Prismatic joint frames: Rx(pi/2) maps joint Z to chassis -Y, so a
    // positive coordinate moves the wheel down from its mount.
    const Mat3 R_susp = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();

    // --- Chassis (body 1) ---
    vm.chassis_body = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
        Transform3::Identity(),
        RigidBodyInertia::from_solid_box(p.chassis_mass, p.chassis_half_extents), "chassis");
    vm.chassis_joint_index = vm.chassis_body;

    // --- Wheel mount positions in chassis frame ---
    const std::array<Vec3, 4> mount_pos = {{
        Vec3( p.front_axle_x, 0.0,  p.half_track),  // FL
        Vec3( p.front_axle_x, 0.0, -p.half_track),  // FR
        Vec3(-p.rear_axle_x,  0.0,  p.half_track),  // RL
        Vec3(-p.rear_axle_x,  0.0, -p.half_track),  // RR
    }};

    const std::array<std::string, 4> names = {{"FL", "FR", "RL", "RR"}};
    const auto I_wheel = RigidBodyInertia::from_solid_box(p.wheel_mass, p.wheel_half_extents);
    const auto prismatic = std::make_shared<kernel::PrismaticJointModel>();

    for (std::size_t c = 0; c < 4; ++c) {
        // --- Wheel body on a prismatic joint from the chassis ---
        vm.wheel_bodies[c] = sys.model.add_body(
            vm.chassis_body, prismatic, Transform3(R_susp, mount_pos[c]),
            Transform3::FromRotation(R_susp), I_wheel, names[c]);
        vm.wheel_joint_indices[c] = vm.wheel_bodies[c];
        vm.susp_q[c] = sys.model.idx_q[vm.wheel_bodies[c]];
        vm.susp_v[c] = sys.model.idx_v[vm.wheel_bodies[c]];

        // --- Suspension spring-damper: chassis mount to wheel origin ---
        sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
            vm.chassis_body, vm.wheel_bodies[c],
            mount_pos[c], Vec3::Zero(),
            p.k_susp, p.c_susp, p.susp_rest_length));

        // --- Tire force ---
        auto tire = std::make_shared<FullTireForce>(
            vm.wheel_bodies[c],
            p.tire_free_radius,
            p.tire_k_z,
            p.tire_c_z,
            p.tire_params);
        vm.tires[c] = tire.get();
        sys.force_elements.push_back(std::move(tire));
    }

    return vm;
}

/// Put the simulator's state at the static equilibrium of the vehicle, at rest.
inline void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleModel& vm)
{
    const auto& p = vm.params;

    // Chassis: centered at equilibrium height, no rotation
    sim.q = sim.system.model.neutral_configuration();
    sim.q(vm.chassis_ty_idx()) = p.chassis_height_eq();

    // Wheels: each at equilibrium suspension extension
    for (std::size_t c = 0; c < 4; ++c) {
        sim.q(vm.susp_q[c]) = p.q_susp_eq();
    }

    sim.v.setZero();
    sim.refresh();
}

} // namespace mbd