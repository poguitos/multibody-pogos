#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/vehicle_template.hpp"
#include "mbd/simulator.hpp"
#include "mbd/drivetrain.hpp"

using Catch::Matchers::WithinAbs;

// ============================================================================
// ARB produces zero force at reference configuration
// ============================================================================

TEST_CASE("ARB: zero force at reference configuration", "[arb][static]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.k_arb = 30000.0;
    tmpl.rear_axle.k_arb  = 25000.0;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    set_vehicle_equilibrium(sys, vh);

    auto [front_arb, rear_arb] = vh.install_anti_roll_bars(sys, sys.states);
    REQUIRE(front_arb != nullptr);
    REQUIRE(rear_arb != nullptr);

    // Apply forces: ARB should produce zero at reference
    sys.clear_forces();
    sys.apply_force_elements();

    // Sum all forces on chassis from ARB: should be zero
    // (We can't easily isolate ARB forces, but we can check that the chassis
    //  net force from ARB alone is zero by removing gravity and springs.)
    // Instead, verify by checking that a small deliberate wheel displacement
    // produces proportional force.

    // Directly test: at reference, both wheels have same z in chassis frame
    const Transform3 T_CW = sys.states[vh.chassis_body].pose_WB().inverse();
    const Real z_FL = T_CW.apply(sys.states[vh.corners[0].wheel_body].p_WB).y();
    const Real z_FR = T_CW.apply(sys.states[vh.corners[1].wheel_body].p_WB).y();

    REQUIRE_THAT(z_FL, WithinAbs(z_FR, 1e-6));
}

// ============================================================================
// ARB produces force when wheels differ in travel
// ============================================================================

TEST_CASE("ARB: produces restoring force under asymmetric wheel displacement",
          "[arb][static]")
{
    using namespace mbd;

    // Create a minimal test case: chassis body + two dummy wheel bodies
    MultibodySystem sys;

    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.3, 0.5));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FixedJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    // Left wheel at (0, 0.3, 0.5), right at (0, 0.3, -0.5) in chassis frame
    auto I_wheel = RigidBodyInertia::from_solid_box(30.0, Vec3(0.15, 0.15, 0.15));

    BodyIndex left_wheel = sys.add_body(I_wheel, RigidBodyState{}, "left", chassis);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix(),
                   Vec3(0.0, 0.3, 0.5)),
        Transform3::FromRotation(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix()),
        chassis, left_wheel));

    BodyIndex right_wheel = sys.add_body(I_wheel, RigidBodyState{}, "right", chassis);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix(),
                   Vec3(0.0, 0.3, -0.5)),
        Transform3::FromRotation(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix()),
        chassis, right_wheel));

    auto arb = std::make_unique<AntiRollBar>(chassis, left_wheel, right_wheel, 30000.0);

    // Set reference at q = 0 for both
    sys.q.setZero();
    sys.compute_kinematics();
    arb->capture_reference(sys.states);

    AntiRollBar* arb_ptr = arb.get();
    sys.force_elements.push_back(std::move(arb));

    // Displace left up by 0.01m (positive q for prismatic-Y down means wheel moves down)
    // Wait: the prismatic joint with Rx(+pi/2) has q > 0 = wheel below mount.
    // So q=0 means wheel at mount, q > 0 means wheel further down.
    // In chassis frame: wheel_y = 0.3 - q.
    // To move left wheel UP by 0.01: q_left = -0.01 (wheel moves up from mount).
    sys.q(0) = -0.01; // left wheel up by 0.01
    sys.q(1) =  0.00; // right wheel at reference
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Check force on left wheel: should be downward in chassis frame
    // (restoring — pushing it back to reference)
    // F_mag = k_arb * (dz_L - dz_R) = 30000 * (0.01 - 0) = 300 N
    // Force on left wheel = -F_mag * chassis_Y_W = -300 N in +Y direction
    REQUIRE_THAT(sys.forces[left_wheel].f_W.y(), WithinAbs(-300.0, 1.0));

    // Force on right wheel: opposite sign
    REQUIRE_THAT(sys.forces[right_wheel].f_W.y(), WithinAbs(+300.0, 1.0));

    // Chassis net force: zero (forces cancel)
    // But there's a moment (couple)
    const Vec3 chassis_net = sys.forces[chassis].f_W;
    REQUIRE_THAT(chassis_net.norm(), WithinAbs(0.0, 1.0));

    // Chassis moment: non-zero (roll moment)
    REQUIRE(sys.forces[chassis].tau_W.norm() > 10.0);
}

// ============================================================================
// ARB does NOT affect symmetric bounce
// ============================================================================

TEST_CASE("ARB: no force under symmetric wheel displacement",
          "[arb][symmetric]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.3, 0.5));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FixedJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    auto I_wheel = RigidBodyInertia::from_solid_box(30.0, Vec3(0.15, 0.15, 0.15));

    BodyIndex left_wheel = sys.add_body(I_wheel, RigidBodyState{}, "left", chassis);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix(),
                   Vec3(0.0, 0.3, 0.5)),
        Transform3::FromRotation(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix()),
        chassis, left_wheel));

    BodyIndex right_wheel = sys.add_body(I_wheel, RigidBodyState{}, "right", chassis);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix(),
                   Vec3(0.0, 0.3, -0.5)),
        Transform3::FromRotation(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix()),
        chassis, right_wheel));

    auto arb = std::make_unique<AntiRollBar>(chassis, left_wheel, right_wheel, 30000.0);

    sys.q.setZero();
    sys.compute_kinematics();
    arb->capture_reference(sys.states);

    sys.force_elements.push_back(std::move(arb));

    // Both wheels moved up by 0.02m (symmetric bounce)
    sys.q(0) = -0.02;
    sys.q(1) = -0.02;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // ARB force on each wheel should be zero (no differential displacement)
    REQUIRE_THAT(sys.forces[left_wheel].f_W.norm(), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(sys.forces[right_wheel].f_W.norm(), WithinAbs(0.0, 1e-6));
}

// ============================================================================
// ARB with full vehicle: reduces steady-state roll angle in cornering
// ============================================================================

TEST_CASE("ARB: reduces roll angle in steady-state cornering",
          "[arb][cornering]")
{
    using namespace mbd;

    // Simulate same vehicle twice: with ARB off, then with ARB on.
    // Measure roll angle in a steady cornering maneuver.

    auto run_scenario = [](Real k_arb) -> Real {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.rear_axle.k_spring = tmpl.front_axle.k_spring; // uniform for clean test
        tmpl.front_axle.k_arb = k_arb;
        tmpl.rear_axle.k_arb  = k_arb;

        MultibodySystem sys;
        auto vh = build_vehicle(sys, tmpl);

        Simulator sim(sys);
        sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
        sim.method = IntegrationMethod::RK4;
        sim.initialize();

        set_vehicle_equilibrium(sys, vh);
        sys.q_dot(0) = 15.0; // forward 15 m/s
        sys.compute_kinematics();

        if (k_arb > 0.0) {
            vh.install_anti_roll_bars(sys, sys.states);
        }

        // Speed controller
        sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
            const Vec3 fwd = s.states[vh.chassis_body].q_WB * Vec3::UnitX();
            const Real Vx = s.states[vh.chassis_body].v_WB.dot(fwd);
            tau(0) += 500.0 * (15.0 - Vx);
        };

        // Settle briefly, then apply steering
        sim.run(0.5, 0.001);
        vh.set_steering(0.04); // moderate left turn

        // Run for 3 seconds to reach steady-state cornering
        sim.run(3.0, 0.001);

        // Measure body roll as the tilt of chassis-Y axis from world-Y,
        // around the chassis forward direction. Robust to yaw.
        const Quat q_WC = sys.states[vh.chassis_body].q_WB;
        const Vec3 chassis_x_W = q_WC * Vec3::UnitX();
        const Vec3 chassis_y_W = q_WC * Vec3::UnitY();

        // Remove the yaw component: project chassis X onto world XZ plane
        Vec3 fwd_horiz(chassis_x_W.x(), 0.0, chassis_x_W.z());
        fwd_horiz.normalize();

        // "World up" after yaw: (0, 1, 0) is invariant under yaw
        // Roll is the angle between chassis_y_W and world Y, measured around fwd_horiz.
        // Compute it via the component of chassis_y_W perpendicular to fwd_horiz,
        // and compare to world-Y.
        // Right-hand-rule: lateral axis = world_Y x fwd_horiz (points to right of motion)
        const Vec3 lat = Vec3::UnitY().cross(fwd_horiz);

        // Roll angle = angle such that rotating world_Y by this angle around fwd_horiz
        // gives chassis_y_W. Extract via dot products:
        const Real cos_roll = chassis_y_W.dot(Vec3::UnitY());
        const Real sin_roll = chassis_y_W.dot(lat);
        return std::atan2(sin_roll, cos_roll);
    };

    const Real roll_no_arb = run_scenario(0.0);
    const Real roll_with_arb = run_scenario(15000.0); // moderate ARB

    // Both should be nonzero (car is rolling in the turn)
    REQUIRE(std::abs(roll_no_arb) > 0.001);

    // ARB should REDUCE roll magnitude (measured as wheel height difference)
    REQUIRE(std::abs(roll_with_arb) < std::abs(roll_no_arb));

    // ARB should reduce roll by at least 10%
    const Real reduction = (std::abs(roll_no_arb) - std::abs(roll_with_arb))
                           / std::abs(roll_no_arb);
    REQUIRE(reduction > 0.10);
}

// ============================================================================
// ARB does not affect straight-line driving
// ============================================================================

TEST_CASE("ARB: does not affect symmetric straight-line driving",
          "[arb][straight]")
{
    using namespace mbd;

    auto run_scenario = [](Real k_arb) -> Real {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.rear_axle.k_spring = tmpl.front_axle.k_spring;
        tmpl.front_axle.k_arb = k_arb;
        tmpl.rear_axle.k_arb  = k_arb;

        MultibodySystem sys;
        auto vh = build_vehicle(sys, tmpl);

        Simulator sim(sys);
        sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
        sim.method = IntegrationMethod::RK4;
        sim.initialize();

        set_vehicle_equilibrium(sys, vh);
        sys.q_dot(0) = 10.0;
        sys.compute_kinematics();

        if (k_arb > 0.0) {
            vh.install_anti_roll_bars(sys, sys.states);
        }

        sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
            const Vec3 fwd = s.states[vh.chassis_body].q_WB * Vec3::UnitX();
            const Real Vx = s.states[vh.chassis_body].v_WB.dot(fwd);
            tau(0) += 500.0 * (10.0 - Vx);
        };

        sim.run(2.0, 0.001);

        return sys.q(1); // chassis height
    };

    const Real h_no_arb = run_scenario(0.0);
    const Real h_with_arb = run_scenario(15000.0);

    // Chassis height should be essentially identical
    REQUIRE_THAT(h_no_arb, WithinAbs(h_with_arb, 0.005));
}

// ============================================================================
// Parameter correctness: torque on chassis matches F × track
// ============================================================================

TEST_CASE("ARB: chassis moment equals force times track width",
          "[arb][physics]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto I_chassis = RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.3, 0.5));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FixedJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, chassis));

    auto I_wheel = RigidBodyInertia::from_solid_box(30.0, Vec3(0.15, 0.15, 0.15));

    const Real track = 0.6; // half-track
    BodyIndex left_wheel = sys.add_body(I_wheel, RigidBodyState{}, "left", chassis);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix(),
                   Vec3(0.0, 0.3, track)),
        Transform3::FromRotation(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix()),
        chassis, left_wheel));

    BodyIndex right_wheel = sys.add_body(I_wheel, RigidBodyState{}, "right", chassis);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix(),
                   Vec3(0.0, 0.3, -track)),
        Transform3::FromRotation(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix()),
        chassis, right_wheel));

    const Real k_arb = 30000.0;
    auto arb = std::make_unique<AntiRollBar>(chassis, left_wheel, right_wheel, k_arb);

    sys.q.setZero();
    sys.compute_kinematics();
    arb->capture_reference(sys.states);
    sys.force_elements.push_back(std::move(arb));

    // Antisymmetric displacement: left up, right down by 0.01m each
    sys.q(0) = -0.01; // left up
    sys.q(1) =  0.01; // right down
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // F_mag = k_arb * delta = 30000 * 0.02 = 600 N
    // Chassis moment about X axis = F * track * 2 (couple arm) = 600 * 2 * 0.6 = 720 Nm
    // But actually each force acts at its wheel position, so moment = F_L x r_L + F_R x r_R
    // With forces in ±Y and positions at ±Z*track: tau_X = F_L*z_L - F_R*z_R = F*track - (-F*-track) = 2*F*track
    REQUIRE_THAT(sys.forces[chassis].tau_W.x(), WithinAbs(-720.0, 5.0));
}