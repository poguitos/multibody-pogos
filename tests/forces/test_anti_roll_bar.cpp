#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

// Anti-roll bars, on the kernel (plan task 2.7).

using Catch::Matchers::WithinAbs;

namespace
{
    /// A pinned chassis with two wheels on prismatic joints below it, at
    /// (0, 0.3, +-half_track) in the chassis frame. A positive coordinate moves
    /// a wheel down from its mount.
    struct ArbRig {
        mbd::kernel::System sys;
        mbd::BodyIndex chassis{0}, left_wheel{0}, right_wheel{0};
    };

    void make_rig(ArbRig& rig, mbd::Real half_track)
    {
        using namespace mbd;
        rig.chassis = rig.sys.model.add_body(
            0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
            Transform3::Identity(), RigidBodyInertia::from_solid_box(1000.0, Vec3(1.0, 0.3, 0.5)),
            "chassis");
        const Mat3 R_down = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();
        const auto I_wheel = RigidBodyInertia::from_solid_box(30.0, Vec3(0.15, 0.15, 0.15));
        const auto prismatic = std::make_shared<kernel::PrismaticJointModel>();
        rig.left_wheel = rig.sys.model.add_body(rig.chassis, prismatic,
                                                Transform3(R_down, Vec3(0.0, 0.3, half_track)),
                                                Transform3::FromRotation(R_down), I_wheel, "left");
        rig.right_wheel = rig.sys.model.add_body(rig.chassis, prismatic,
                                                 Transform3(R_down, Vec3(0.0, 0.3, -half_track)),
                                                 Transform3::FromRotation(R_down), I_wheel, "right");
    }

    /// Install an ARB with its reference at q = 0, then move the wheels to
    /// (q_left, q_right) and return the body forces there.
    std::vector<mbd::RigidBodyForces> arb_forces(ArbRig& rig, mbd::Real k_arb,
                                                 mbd::Real q_left, mbd::Real q_right)
    {
        using namespace mbd;
        kernel::Simulator sim(rig.sys);   // at q = 0
        auto arb = std::make_shared<AntiRollBar>(rig.chassis, rig.left_wheel, rig.right_wheel, k_arb);
        arb->capture_reference(sim.states());
        rig.sys.force_elements.push_back(arb);

        sim.q(rig.sys.model.idx_q[rig.left_wheel]) = q_left;
        sim.q(rig.sys.model.idx_q[rig.right_wheel]) = q_right;
        sim.acceleration(sim.q, sim.v, 0.0);   // applies the force elements
        return sim.forces();
    }

    mbd::Real forward_speed(const mbd::kernel::Simulator& sim, mbd::BodyIndex chassis)
    {
        const auto& s = sim.states()[static_cast<std::size_t>(chassis)];
        return s.v_WB.dot(s.q_WB * mbd::Vec3::UnitX());
    }
}

// ============================================================================
// ARB produces zero force at reference configuration
// ============================================================================

TEST_CASE("ARB: zero force at reference configuration", "[arb][static]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.k_arb = 30000.0;
    tmpl.rear_axle.k_arb  = 25000.0;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);

    auto [front_arb, rear_arb] = vh.install_anti_roll_bars(sys, sim.states());
    REQUIRE(front_arb != nullptr);
    REQUIRE(rear_arb != nullptr);

    // At the reference, both front wheels are at the same height in the
    // chassis frame, so the bars are untwisted.
    const Transform3 T_CW = sim.states()[static_cast<std::size_t>(vh.chassis_body)].pose_WB().inverse();
    const Real z_FL = T_CW.apply(sim.states()[static_cast<std::size_t>(vh.corners[0].wheel_body)].p_WB).y();
    const Real z_FR = T_CW.apply(sim.states()[static_cast<std::size_t>(vh.corners[1].wheel_body)].p_WB).y();
    REQUIRE_THAT(z_FL, WithinAbs(z_FR, 1e-6));

    // And the bars alone apply nothing.
    std::vector<RigidBodyForces> forces(static_cast<std::size_t>(sys.model.nbodies()));
    front_arb->apply(sim.states(), forces);
    rear_arb->apply(sim.states(), forces);
    for (const auto& f : forces) {
        REQUIRE_THAT(f.f_W.norm(), WithinAbs(0.0, 1e-9));
    }
}

// ============================================================================
// ARB produces force when wheels differ in travel
// ============================================================================

TEST_CASE("ARB: produces restoring force under asymmetric wheel displacement",
          "[arb][static]")
{
    using namespace mbd;

    ArbRig rig;
    make_rig(rig, 0.5);

    // Left wheel up by 0.01 m (a negative coordinate raises it), right at reference
    const auto forces = arb_forces(rig, 30000.0, -0.01, 0.0);

    // F_mag = k_arb * (dz_L - dz_R) = 30000 * (0.01 - 0) = 300 N, restoring:
    // down on the left wheel, up on the right.
    REQUIRE_THAT(forces[static_cast<std::size_t>(rig.left_wheel)].f_W.y(), WithinAbs(-300.0, 1.0));
    REQUIRE_THAT(forces[static_cast<std::size_t>(rig.right_wheel)].f_W.y(), WithinAbs(+300.0, 1.0));

    // Chassis net force: zero (forces cancel), but there's a moment (couple)
    REQUIRE_THAT(forces[static_cast<std::size_t>(rig.chassis)].f_W.norm(), WithinAbs(0.0, 1.0));
    REQUIRE(forces[static_cast<std::size_t>(rig.chassis)].tau_W.norm() > 10.0);
}

// ============================================================================
// ARB does NOT affect symmetric bounce
// ============================================================================

TEST_CASE("ARB: no force under symmetric wheel displacement",
          "[arb][symmetric]")
{
    using namespace mbd;

    ArbRig rig;
    make_rig(rig, 0.5);

    // Both wheels moved up by 0.02m (symmetric bounce)
    const auto forces = arb_forces(rig, 30000.0, -0.02, -0.02);

    // ARB force on each wheel should be zero (no differential displacement)
    REQUIRE_THAT(forces[static_cast<std::size_t>(rig.left_wheel)].f_W.norm(), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(forces[static_cast<std::size_t>(rig.right_wheel)].f_W.norm(), WithinAbs(0.0, 1e-6));
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

        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);

        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);
        const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;
        sim.v(v_forward) = 15.0; // forward 15 m/s
        sim.initialize();

        if (k_arb > 0.0) {
            vh.install_anti_roll_bars(sys, sim.states());
        }

        // Speed controller
        sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
            tau(v_forward) += 500.0 * (15.0 - forward_speed(s, vh.chassis_body));
        };

        // Settle briefly, then apply steering
        sim.run(0.5, 0.001);
        vh.set_steering(0.04); // moderate left turn

        // Run for 3 seconds to reach steady-state cornering
        sim.run(3.0, 0.001);

        // Measure body roll as the tilt of chassis-Y axis from world-Y,
        // around the chassis forward direction. Robust to yaw.
        const Quat q_WC = sim.states()[static_cast<std::size_t>(vh.chassis_body)].q_WB;
        const Vec3 chassis_x_W = q_WC * Vec3::UnitX();
        const Vec3 chassis_y_W = q_WC * Vec3::UnitY();

        // Remove the yaw component: project chassis X onto world XZ plane
        Vec3 fwd_horiz(chassis_x_W.x(), 0.0, chassis_x_W.z());
        fwd_horiz.normalize();

        // Lateral axis = world_Y x fwd_horiz (points to the right of the motion);
        // roll is the angle of chassis Y from world Y about the forward axis.
        const Vec3 lat = Vec3::UnitY().cross(fwd_horiz);
        const Real cos_roll = chassis_y_W.dot(Vec3::UnitY());
        const Real sin_roll = chassis_y_W.dot(lat);
        return std::atan2(sin_roll, cos_roll);
    };

    const Real roll_no_arb = run_scenario(0.0);
    const Real roll_with_arb = run_scenario(15000.0); // moderate ARB

    // Both should be nonzero (car is rolling in the turn)
    REQUIRE(std::abs(roll_no_arb) > 0.001);

    // ARB should REDUCE roll magnitude
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

        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);

        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);
        const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;
        sim.v(v_forward) = 10.0;
        sim.initialize();

        if (k_arb > 0.0) {
            vh.install_anti_roll_bars(sys, sim.states());
        }

        sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
            tau(v_forward) += 500.0 * (10.0 - forward_speed(s, vh.chassis_body));
        };

        sim.run(2.0, 0.001);

        return sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.y(); // chassis height
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

    const Real track = 0.6; // half-track
    ArbRig rig;
    make_rig(rig, track);

    // Antisymmetric displacement: left up, right down by 0.01m each
    const auto forces = arb_forces(rig, 30000.0, -0.01, 0.01);

    // F_mag = k_arb * delta = 30000 * 0.02 = 600 N
    // With forces in ±Y and positions at ±Z*track: tau_X = F_L*z_L - F_R*z_R = 2*F*track
    REQUIRE_THAT(forces[static_cast<std::size_t>(rig.chassis)].tau_W.x(), WithinAbs(-720.0, 5.0));
}
