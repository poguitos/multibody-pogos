#pragma once

// Simplified full vehicle model builder.
//
// Creates a 5-body, 10-DOF vehicle: chassis on FreeCoordJoint + 4 wheels
// on PrismaticCoordJoints for vertical suspension travel.
//
// q layout: [tx, ty, tz, rx, ry, rz, q_FL, q_FR, q_RL, q_RR]
//   0-2: chassis translation (in joint frame ≈ world at small angles)
//   3-5: chassis rotation (exponential map)
//   6-9: suspension travel per corner (positive = wheel moves down from mount)

#include "mbd/system.hpp"
#include "mbd/tire.hpp"

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
    std::array<FullTireForce*, 4> tires{};
    VehicleParams params;

    /// Index of the chassis joint coordinate q_dot(1) (ty, vertical)
    int chassis_ty_idx() const { return 1; }

    /// Index of a wheel's suspension q in the global q vector.
    int susp_q_idx(Corner c) const
    {
        return 6 + static_cast<int>(c);
    }
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

/// Build a simplified vehicle MultibodySystem.
/// Returns a VehicleModel with indices for easy access.
inline VehicleModel build_simple_vehicle(MultibodySystem& sys,
                                         const VehicleParams& p = VehicleParams{})
{
    VehicleModel vm;
    vm.params = p;

    // Prismatic joint rotation: Rx(pi/2) maps joint Z → parent -Y (downward).
    // Wait — Rx(π/2): x→x, y→-z, z→y. So joint Z → parent +Y.
    // We want q>0 = downward, so we need joint Z → parent -Y.
    // Use Rx(-π/2): x→x, y→z, z→-y. Joint Z → parent -Y. Correct.
    const Mat3 R_susp = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    // --- Chassis (body 1) ---
    auto I_chassis = RigidBodyInertia::from_solid_box(
        p.chassis_mass, p.chassis_half_extents);
    vm.chassis_body = sys.add_body(
        I_chassis, RigidBodyState{}, "chassis", kGroundIndex);

    auto chassis_joint = std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, vm.chassis_body);
    vm.chassis_joint_index = sys.add_joint(std::move(chassis_joint));

    // --- Wheel mount positions in chassis frame ---
    const std::array<Vec3, 4> mount_pos = {{
        Vec3( p.front_axle_x, 0.0,  p.half_track),  // FL
        Vec3( p.front_axle_x, 0.0, -p.half_track),  // FR
        Vec3(-p.rear_axle_x,  0.0,  p.half_track),  // RL
        Vec3(-p.rear_axle_x,  0.0, -p.half_track),  // RR
    }};

    const std::array<std::string, 4> names = {{"FL", "FR", "RL", "RR"}};

    auto I_wheel = RigidBodyInertia::from_solid_box(
        p.wheel_mass, p.wheel_half_extents);

    for (int c = 0; c < 4; ++c) {
        // --- Add wheel body ---
        vm.wheel_bodies[c] = sys.add_body(
            I_wheel, RigidBodyState{}, names[c], vm.chassis_body);

        // --- Prismatic joint from chassis to wheel ---
        Transform3 X_PJ(R_susp, mount_pos[c]);
        Transform3 X_CJ = Transform3::FromRotation(R_susp);

        auto joint = std::make_unique<PrismaticCoordJoint>(
            X_PJ, X_CJ, vm.chassis_body, vm.wheel_bodies[c]);
        vm.wheel_joint_indices[c] = sys.add_joint(std::move(joint));

        // --- Suspension spring-damper ---
        // Connects chassis mount point to wheel body origin.
        // anchor1 on chassis = mount_pos[c] (chassis body frame)
        // anchor2 on wheel = (0, 0, 0) (wheel body frame)
        sys.force_elements.push_back(std::make_unique<LinearSpringDamper>(
            vm.chassis_body, vm.wheel_bodies[c],
            mount_pos[c], Vec3::Zero(),
            p.k_susp, p.c_susp, p.susp_rest_length));

        // --- Tire force ---
        auto tire = std::make_unique<FullTireForce>(
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

/// Set the vehicle to static equilibrium initial conditions.
inline void set_vehicle_equilibrium(MultibodySystem& sys, const VehicleModel& vm)
{
    const auto& p = vm.params;

    // Chassis: centered at equilibrium height, no rotation
    sys.q.setZero();
    sys.q(1) = p.chassis_height_eq(); // ty = CG height

    // Wheels: each at equilibrium suspension extension
    for (int c = 0; c < 4; ++c) {
        sys.q(6 + c) = p.q_susp_eq();
    }

    sys.q_dot.setZero();
    sys.compute_kinematics();
}

} // namespace mbd