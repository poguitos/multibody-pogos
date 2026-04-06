#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle.hpp"
#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }
}

// ============================================================================
// Ackermann geometry (pure math, no simulation)
// ============================================================================

TEST_CASE("Steering: Ackermann at zero returns zero", "[steering][ackermann]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vm = build_simple_vehicle(sys);

    auto [fl, fr] = vm.ackermann_steering(0.0);
    REQUIRE_THAT(fl, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(fr, WithinAbs(0.0, 1e-12));
}

TEST_CASE("Steering: Ackermann left turn inner angle larger than outer",
          "[steering][ackermann]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vm = build_simple_vehicle(sys);

    const Real delta = 0.1;
    auto [fl, fr] = vm.ackermann_steering(delta);

    // Both positive (left turn)
    REQUIRE(fl > 0.0);
    REQUIRE(fr > 0.0);

    // FL is inner wheel for left turn: larger angle
    REQUIRE(fl > fr);

    // Average should be close to input
    const Real avg = 0.5 * (fl + fr);
    REQUIRE_THAT(avg, WithinAbs(delta, delta * 0.05));
}

TEST_CASE("Steering: Ackermann right turn signs flip correctly",
          "[steering][ackermann]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vm = build_simple_vehicle(sys);

    const Real delta = -0.1;
    auto [fl, fr] = vm.ackermann_steering(delta);

    // Both negative
    REQUIRE(fl < 0.0);
    REQUIRE(fr < 0.0);

    // FR is inner wheel for right turn: |FR| > |FL|
    REQUIRE(std::abs(fr) > std::abs(fl));
}

TEST_CASE("Steering: Ackermann is antisymmetric",
          "[steering][ackermann]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vm = build_simple_vehicle(sys);

    auto [fl_L, fr_L] = vm.ackermann_steering(0.08);
    auto [fl_R, fr_R] = vm.ackermann_steering(-0.08);

    REQUIRE_THAT(fl_L, WithinAbs(-fr_R, 1e-10));
    REQUIRE_THAT(fr_L, WithinAbs(-fl_R, 1e-10));
}

// ============================================================================
// Tire-level steering effect
// ============================================================================

TEST_CASE("Steering: steered tire develops lateral force from pure forward velocity",
          "[steering][tire]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vm = build_simple_vehicle(sys);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    // Give the vehicle forward speed
    sys.q_dot(0) = 15.0;
    sys.compute_kinematics();

    // No steering: FL tire should have near-zero Fy
    sys.clear_forces();
    sys.apply_force_elements();
    const Real Fy_no_steer = vm.tires[0]->get_Fy();
    REQUIRE_THAT(Fy_no_steer, WithinAbs(0.0, 10.0));

    // Apply 0.05 rad left steering to FL tire only
    vm.tires[0]->steer_angle = 0.05;

    sys.clear_forces();
    sys.apply_force_elements();
    const Real Fy_steered = vm.tires[0]->get_Fy();

    // Steered tire should now produce significant lateral force
    REQUIRE(std::abs(Fy_steered) > 500.0);

    // Clean up
    vm.tires[0]->steer_angle = 0.0;
}

// ============================================================================
// Vehicle cornering simulation
// ============================================================================

TEST_CASE("Steering: vehicle turns left with positive steering angle",
          "[steering][cornering]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    // Start already at speed to avoid violent transient
    sys.q_dot(0) = 10.0;
    sys.compute_kinematics();

    const Real delta_steer = 0.03;
    const Real K_speed = 500.0;
    const Real V_target = 10.0;

    vm.set_front_steering(delta_steer);

    sim.force_callback = [&](MultibodySystem& s, Real /*t*/, VecX& tau) {
        const Vec3 fwd_W = s.states[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vm.chassis_body].v_WB.dot(fwd_W);
        tau(0) += K_speed * (V_target - Vx);
    };

    // Let suspension settle for 0.5s, then corner for 3s
    vm.clear_steering();
    sim.run(0.5, 0.001);

    vm.set_front_steering(delta_steer);
    const Real z_before = sys.states[vm.chassis_body].p_WB.z();
    sim.run(3.0, 0.001);
    const Real z_after = sys.states[vm.chassis_body].p_WB.z();

    // Vehicle should have moved LEFT (positive Z)
    REQUIRE(z_after - z_before > 0.05);

    // Vehicle should still be near ground
    REQUIRE_THAT(sys.states[vm.chassis_body].p_WB.y(),
                 WithinAbs(params.chassis_height_eq(), 0.05));
}

TEST_CASE("Steering: vehicle turns right with negative steering angle",
          "[steering][cornering]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    sys.q_dot(0) = 10.0;
    sys.compute_kinematics();

    const Real delta_steer = -0.03;
    const Real K_speed = 500.0;
    const Real V_target = 10.0;

    sim.force_callback = [&](MultibodySystem& s, Real /*t*/, VecX& tau) {
        const Vec3 fwd_W = s.states[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vm.chassis_body].v_WB.dot(fwd_W);
        tau(0) += K_speed * (V_target - Vx);
    };

    // Settle first
    sim.run(0.5, 0.001);

    vm.set_front_steering(delta_steer);
    const Real z_before = sys.states[vm.chassis_body].p_WB.z();
    sim.run(3.0, 0.001);
    const Real z_after = sys.states[vm.chassis_body].p_WB.z();

    // Vehicle should have moved RIGHT (negative Z)
    REQUIRE(z_after - z_before < -0.05);
}

// ============================================================================
// Straight-line regression
// ============================================================================

TEST_CASE("Steering: zero steering produces straight-line motion",
          "[steering][straight]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    vm.clear_steering();

    sys.q_dot(0) = 10.0;
    sys.compute_kinematics();

    const Real K_speed = 500.0;
    const Real V_target = 10.0;

    sim.force_callback = [&](MultibodySystem& s, Real /*t*/, VecX& tau) {
        const Vec3 fwd_W = s.states[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vm.chassis_body].v_WB.dot(fwd_W);
        tau(0) += K_speed * (V_target - Vx);
    };

    sim.run(0.5, 0.001); // settle
    sim.run(2.0, 0.001);

    // Vehicle moves forward
    REQUIRE(sys.states[vm.chassis_body].p_WB.x() > 10.0);

    // No lateral displacement
    REQUIRE_THAT(sys.states[vm.chassis_body].p_WB.z(), WithinAbs(0.0, 0.02));

    // No yaw
    const Vec3 fwd_W = sys.states[vm.chassis_body].q_WB * Vec3::UnitX();
    REQUIRE_THAT(fwd_W.z(), WithinAbs(0.0, 0.005));
}

// ============================================================================
// Low-speed turn radius vs kinematic prediction
// ============================================================================

TEST_CASE("Steering: low-speed turn radius approximately matches kinematic prediction",
          "[steering][radius]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    sys.q_dot(0) = 8.0;
    sys.compute_kinematics();

    const Real V_target = 8.0;
    const Real delta = 0.02;
    const Real K_speed = 500.0;

    sim.force_callback = [&](MultibodySystem& s, Real /*t*/, VecX& tau) {
        const Vec3 fwd_W = s.states[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vm.chassis_body].v_WB.dot(fwd_W);
        tau(0) += K_speed * (V_target - Vx);
    };

    // Settle suspension at speed without steering
    sim.run(1.0, 0.001);

    // Apply steering and corner for 4 seconds
    vm.set_front_steering(delta);
    sim.run(4.0, 0.001);

    // Kinematic turn radius
    const Real L = params.front_axle_x + params.rear_axle_x;
    const Real R_kinematic = L / std::tan(delta);

    // Measure actual turn radius from yaw rate
    const Real omega_yaw = sys.states[vm.chassis_body].w_WB.y();
    const Vec3 fwd_W = sys.states[vm.chassis_body].q_WB * Vec3::UnitX();
    const Real V_actual = sys.states[vm.chassis_body].v_WB.dot(fwd_W);

    REQUIRE(std::abs(omega_yaw) > 0.001);
    const Real R_actual = V_actual / std::abs(omega_yaw);

    // At low speed, should match within 20%
    const Real rel_error = std::abs(R_actual - R_kinematic) / R_kinematic;
    REQUIRE(rel_error < 0.20);
}

// ============================================================================
// set_front_steering applies to both front tires
// ============================================================================

TEST_CASE("Steering: set_front_steering applies Ackermann to front tires only",
          "[steering][interface]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vm = build_simple_vehicle(sys);

    vm.set_front_steering(0.1);

    // Front tires should have nonzero (and different) angles
    REQUIRE(vm.tires[0]->steer_angle > 0.0);
    REQUIRE(vm.tires[1]->steer_angle > 0.0);
    REQUIRE(vm.tires[0]->steer_angle > vm.tires[1]->steer_angle);

    // Rear tires unchanged
    REQUIRE_THAT(vm.tires[2]->steer_angle, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(vm.tires[3]->steer_angle, WithinAbs(0.0, 1e-12));

    // clear_steering resets everything
    vm.clear_steering();
    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(vm.tires[c]->steer_angle, WithinAbs(0.0, 1e-12));
    }
}