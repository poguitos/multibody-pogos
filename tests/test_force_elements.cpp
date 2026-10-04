#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include <mbd/forces/force_element.hpp>

// The spring-damper force element, given body states directly.

using Catch::Matchers::WithinAbs;

TEST_CASE("LinearSpringDamper applies correct force between ground and body",
          "[force_element]")
{
    using namespace mbd;

    RigidBodyState s_body;
    s_body.p_WB = Vec3(2.0, 0.0, 0.0);
    s_body.q_WB = Quat(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitZ()));
    s_body.v_WB = Vec3(1.0, 0.0, 0.0);
    s_body.w_WB = Vec3::Zero();

    const BodyIndex b_body = 1;
    std::vector<RigidBodyState> states(2);
    states[1] = s_body;
    std::vector<RigidBodyForces> forces(2);

    // Spring from ground anchor at (0,1,0) to body-local anchor at (0,0.5,0)
    LinearSpringDamper spring(kGroundIndex, b_body,
                              Vec3(0.0, 1.0, 0.0),
                              Vec3(0.0, 0.5, 0.0),
                              100.0, 10.0, 1.0);

    spring.apply(states, forces);

    auto& forces_body = forces[b_body];

    // Body rotated 90 deg about Z: local Y -> -world X
    // Body attachment in world: (2,0,0) + Rz(90)*(0,0.5,0) = (2,0,0) + (-0.5,0,0) = (1.5,0,0)
    // Ground anchor in world: (0,1,0) (ground is identity pose)
    // diff = (1.5, -1.0, 0)
    // Force pulls body attachment toward ground anchor:
    //   x-component negative (pull left), y-component positive (pull up)
    REQUIRE(forces_body.f_W.x() < -0.01);
    REQUIRE(forces_body.f_W.y() > 0.01);

    // Torque: lever arm r = (-0.5, 0, 0), force has +Y component
    // r cross F: (-0.5,0,0) x (Fx,Fy,0) = (0, 0, -0.5*Fy)  =>  tau_z < 0
    REQUIRE(forces_body.tau_W.z() < -0.001);

    // Newton's 3rd law: ground forces equal and opposite
    auto& forces_ground = forces[kGroundIndex];
    REQUIRE_THAT(forces_ground.f_W.x() + forces_body.f_W.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(forces_ground.f_W.y() + forces_body.f_W.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(forces_ground.f_W.z() + forces_body.f_W.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("LinearSpringDamper at rest length with zero velocity gives zero force",
          "[force_element]")
{
    using namespace mbd;

    RigidBodyState s_body;
    s_body.p_WB = Vec3(1.0, 0.0, 0.0);
    const BodyIndex b_body = 1;
    std::vector<RigidBodyState> states(2);
    states[1] = s_body;
    std::vector<RigidBodyForces> forces(2);

    // Spring from ground origin to body origin, rest_length = 1.0, no velocity
    LinearSpringDamper spring(kGroundIndex, b_body,
                              Vec3::Zero(), Vec3::Zero(),
                              100.0, 10.0, 1.0);

    spring.apply(states, forces);

    REQUIRE_THAT(forces[b_body].f_W.norm(), WithinAbs(0.0, 1e-9));
}