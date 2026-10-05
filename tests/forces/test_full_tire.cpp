#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include <memory>
#include <utility>
#include <vector>

#include "mbd/forces/tire.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/spatial/spatial.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-9;

    /// A single wheel for testing the tyre force element, which reads only the
    /// wheel's world state. q holds the position and a rotation vector, q_dot
    /// the world velocity of the wheel centre (its last three entries, rotation
    /// rates, must stay zero: no test spins the wheel body).
    struct FreeWheel {
        Eigen::Matrix<mbd::Real, 6, 1> q = Eigen::Matrix<mbd::Real, 6, 1>::Zero();
        Eigen::Matrix<mbd::Real, 6, 1> q_dot = Eigen::Matrix<mbd::Real, 6, 1>::Zero();
        std::vector<mbd::RigidBodyState> states = std::vector<mbd::RigidBodyState>(2);
        std::vector<mbd::RigidBodyForces> forces = std::vector<mbd::RigidBodyForces>(2);
        std::shared_ptr<mbd::FullTireForce> tire;

        void compute_kinematics()
        {
            REQUIRE(q_dot.tail<3>().isZero());
            states[1] = mbd::RigidBodyState(q.head<3>(), mbd::exp3(q.tail<3>()), q_dot.head<3>(),
                                            mbd::Vec3::Zero());
        }
        void clear_forces() { forces.assign(2, mbd::RigidBodyForces{}); }
        void apply_force_elements()
        {
            compute_kinematics();
            tire->apply(states, forces);
        }
    };

    /// A wheel with a FullTireForce. Returns {wheel, tire_ptr}.
    std::pair<FreeWheel, mbd::FullTireForce*>
    make_free_wheel(mbd::Real /*mass*/, mbd::Real R_free, mbd::Real k_z, mbd::Real c_z)
    {
        using namespace mbd;
        FreeWheel w;
        w.tire = std::make_shared<FullTireForce>(1, R_free, k_z, c_z,
                                                 PacejkaTireParams::DefaultPassengerCar());
        FullTireForce* ptr = w.tire.get();
        return {std::move(w), ptr};
    }
}

// ============================================================================
// Vertical contact behavior
// ============================================================================

TEST_CASE("FullTireForce: no contact when wheel above ground",
          "[full_tire][vertical]")
{
    using namespace mbd;

    auto [sys, tire] = make_free_wheel(40.0, 0.35, 200000.0, 500.0);

    // Wheel at y = 0.5 (well above R_free = 0.35 + ground at 0)
    sys.q << 0.0, 0.5, 0.0, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(0.0, eps));
    REQUIRE_THAT(tire->get_Fx(), WithinAbs(0.0, eps));
    REQUIRE_THAT(tire->get_Fy(), WithinAbs(0.0, eps));
}

TEST_CASE("FullTireForce: vertical force at static deflection",
          "[full_tire][vertical]")
{
    using namespace mbd;

    const Real mass = 40.0;
    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(mass, R_free, k_z, 500.0);

    // Wheel at R_free - 0.01 (10mm compression)
    sys.q << 0.0, R_free - 0.01, 0.0, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Fz = k_z * 0.01 = 2000 N
    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(2000.0, 1.0));
    REQUIRE_THAT(tire->get_deflection(), WithinAbs(0.01, 1e-6));
}

// ============================================================================
// Slip angle from lateral velocity
// ============================================================================

TEST_CASE("FullTireForce: lateral velocity produces slip angle and Fy",
          "[full_tire][lateral]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<mbd::FullTireForce*>(tire_ptr);
    tire->auto_free_roll = false;
    // 20mm compression → Fz ≈ 4000 N
    const Real defl = 0.02;
    sys.q << 0.0, R_free - defl, 0.0, 0.0, 0.0, 0.0;

    // Forward 20 m/s, lateral 1 m/s
    sys.q_dot << 20.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    // Set omega to free-rolling so kappa ≈ 0 (otherwise combined slip kills Fy)
    const Real R_eff = R_free - defl * 0.5;
    tire->omega_wheel = 20.0 / R_eff;

    sys.clear_forces();
    sys.apply_force_elements();

    // Should have contact
    REQUIRE(tire->get_vertical_force() > 100.0);

    // Slip angle should be nonzero
    REQUIRE(std::abs(tire->get_slip_angle()) > 0.01);

    // Fy should be nonzero
    REQUIRE(std::abs(tire->get_Fy()) > 100.0);

    // At this low slip angle, force should be approximately K_alpha * alpha
    const Real Fz_actual = tire->get_vertical_force();
    const Real alpha_actual = tire->get_slip_angle();
    const Real K_alpha = tire->pacejka.cornering_stiffness(Fz_actual);
    const Real Fy_linear = K_alpha * alpha_actual;

    // Should agree within 20% (we're in the linear region)
    const Real rel_error = std::abs(tire->get_Fy() - Fy_linear) /
                           std::max(std::abs(Fy_linear), Real(1.0));
    REQUIRE(rel_error < 0.2);
}
// ============================================================================
// Slip ratio from forward velocity (locked wheel)
// ============================================================================

TEST_CASE("FullTireForce: forward velocity with locked wheel produces slip ratio",
          "[full_tire][longitudinal]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<FullTireForce*>(tire_ptr);
    tire->auto_free_roll = false;
    tire->omega_wheel = 0.0;

    const Real defl = 0.02;
    sys.q << 0.0, R_free - defl, 0.0, 0.0, 0.0, 0.0;

    // Moving forward at 20 m/s, wheel not spinning (locked wheel)
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Slip ratio should be strongly negative (locked wheel braking)
    REQUIRE(tire->get_slip_ratio() < -0.5);

    // Fx should be negative (braking force, opposing forward motion)
    REQUIRE(tire->get_Fx() < -100.0);
}

TEST_CASE("FullTireForce: free-rolling wheel has near-zero slip ratio",
          "[full_tire][longitudinal]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<mbd::FullTireForce*>(tire_ptr);
    tire->auto_free_roll = false;

    const Real defl = 0.02;
    sys.q << 0.0, R_free - defl, 0.0, 0.0, 0.0, 0.0;

    const Real Vx = 20.0;
    sys.q_dot << Vx, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    // Set omega to match free-rolling: omega = Vx / R_eff
    const Real R_eff = R_free - defl * 0.5;
    tire->omega_wheel = Vx / R_eff;

    sys.clear_forces();
    sys.apply_force_elements();

    // Slip ratio should be near zero
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(0.0, 0.02));

    // Fx should be near zero
    REQUIRE_THAT(tire->get_Fx(), WithinAbs(0.0, 50.0));
}

// ============================================================================
// Combined slip
// ============================================================================

TEST_CASE("FullTireForce: combined slip produces both Fx and Fy",
          "[full_tire][combined]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<FullTireForce*>(tire_ptr);
    tire->auto_free_roll = false;

    const Real defl = 0.02;
    sys.q << 0.0, R_free - defl, 0.0, 0.0, 0.0, 0.0;

    // Forward at 20 m/s, lateral at 2 m/s, partial braking
    sys.q_dot << 20.0, 0.0, 2.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    const Real R_eff = R_free - defl * 0.5;
    tire->omega_wheel = 15.0 / R_eff; // Under free-rolling speed = braking

    sys.clear_forces();
    sys.apply_force_elements();

    // Both Fx and Fy should be nonzero
    REQUIRE(std::abs(tire->get_Fx()) > 10.0);
    REQUIRE(std::abs(tire->get_Fy()) > 10.0);

    // Combined Fy should be less than pure lateral Fy at same alpha
    const Real alpha = tire->get_slip_angle();
    const Real Fz = tire->get_vertical_force();
    auto pure_result = tire->pacejka.compute(0.0, alpha, Fz);
    const Real Fy_pure_abs = std::abs(pure_result.Fy) + 1.0;
    REQUIRE(std::abs(tire->get_Fy()) < Fy_pure_abs);
}

// ============================================================================
// Static settling with FullTireForce
// ============================================================================

TEST_CASE("FullTireForce: free wheel settles to static equilibrium under gravity",
          "[full_tire][settling]")
{
    using namespace mbd;

    const Real mass = 40.0;
    const Real R_free = 0.35;
    const Real k_z = 200000.0;
    const Real c_z = 2000.0;

    // A free wheel body on the kernel, with the tyre as its only force.
    kernel::System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    sys.model.add_body(0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
                       Transform3::Identity(),
                       RigidBodyInertia::from_solid_box(mass, Vec3(0.15, R_free, 0.15)), "wheel");
    sys.force_elements.push_back(std::make_shared<FullTireForce>(
        1, R_free, k_z, c_z, PacejkaTireParams::DefaultPassengerCar()));

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    // Start slightly above equilibrium
    const Real y_eq = R_free - mass * g_accel / k_z;
    sim.q(1) = y_eq + 0.02;
    sim.initialize();

    sim.run(2.0, 0.001);

    // Should settle near equilibrium, without moving horizontally
    const RigidBodyState& wheel = sim.states()[1];
    REQUIRE_THAT(wheel.p_WB.y(), WithinAbs(y_eq, 0.002));
    REQUIRE_THAT(wheel.p_WB.x(), WithinAbs(0.0, 0.001));
    REQUIRE_THAT(wheel.p_WB.z(), WithinAbs(0.0, 0.001));

    // Velocities near zero
    for (int i = 0; i < 6; ++i) {
        REQUIRE_THAT(sim.v(i), WithinAbs(0.0, 0.05));
    }
}

// ============================================================================
// Force directions
// ============================================================================

TEST_CASE("FullTireForce: rotated wheel produces correct force directions",
          "[full_tire][rotated]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);

    const Real defl = 0.02;

    // Wheel yawed 90 degrees about Y: body X now points along world -Z.
    // Tire forward direction in world = (0, 0, -1).
    sys.q << 0.0, R_free - defl, 0.0, 0.0, pi / 2.0, 0.0;

    // We want the wheel to move along world +X (perpendicular to tire forward).
    // FreeWheel's q_dot(0:2) is the world velocity of the wheel centre, so
    // q_dot(0) = 20 gives 20 m/s along world X directly.
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Tire moves entirely sideways: Vx ≈ 0, Vy ≈ 20 → large slip angle
    REQUIRE(std::abs(tire->get_slip_angle()) > 0.5);
    REQUIRE(std::abs(tire->get_Fy()) > 500.0);

    // Tire forward is world -Z, lateral is world -X (from Y x forward).
    // Fy acts along the lateral direction, which has a world X component.
    // Vertical force acts along world Y.
    // So total force should have significant components in X and/or Z plane.
    const Real lateral_force_magnitude =
        std::sqrt(sys.forces[1].f_W.x() * sys.forces[1].f_W.x() +
                  sys.forces[1].f_W.z() * sys.forces[1].f_W.z());
    REQUIRE(lateral_force_magnitude > 100.0);
}

// ============================================================================
// Telemetry accessors
// ============================================================================

TEST_CASE("FullTireForce: telemetry matches internal state",
          "[full_tire][telemetry]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);

    const Real defl = 0.02;
    sys.q << 0.0, R_free - defl, 0.0, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // All telemetry should be populated
    REQUIRE(tire->get_vertical_force() > 0.0);
    REQUIRE(tire->get_deflection() > 0.0);

    // Result struct should match accessors
    const auto& r = tire->get_last_result();
    REQUIRE_THAT(r.Fx, WithinAbs(tire->get_Fx(), eps));
    REQUIRE_THAT(r.Fy, WithinAbs(tire->get_Fy(), eps));
    REQUIRE_THAT(r.kappa, WithinAbs(tire->get_slip_ratio(), eps));
    REQUIRE_THAT(r.alpha, WithinAbs(tire->get_slip_angle(), eps));
    REQUIRE_THAT(r.Fz, WithinAbs(tire->get_vertical_force(), eps));
}