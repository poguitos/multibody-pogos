#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mbd/force_element.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("LinearSpringDamper applies correct force and torque", "[force_element]")
{
    using namespace mbd;

    // Setup:
    // Spring connected to World Origin (0,0,0)
    // Connected to Body at (0, 0.5, 0) relative to COM
    // Body COM is at (2, 0, 0)
    // Body is rotated 90 deg around Z.
    
    // 1. Define geometry
    Vec3 anchor_W(0.0, 0.0, 0.0);
    Vec3 anchor_B(0.0, 0.5, 0.0); // Offset in Y_body
    Real k = 100.0;
    Real c = 10.0;
    Real rest_length = 1.0;
    
    LinearSpringDamper spring(anchor_W, anchor_B, k, c, rest_length);

    // 2. Set State
    RigidBodyState state;
    state.p_WB = Vec3(2.0, 0.0, 0.0); // COM at x=2
    // Rotate 90 deg around Z: X_body -> Y_world, Y_body -> -X_world
    state.q_WB = Quat(Eigen::AngleAxisd(mbd::pi/2.0, Vec3::UnitZ()));
    state.v_WB = Vec3(1.0, 0.0, 0.0); // Moving away from origin at 1 m/s
    state.w_WB = Vec3::Zero();

    // 3. Expected Geometry Analysis:
    // R_WB * anchor_B 
    // R_z(90) * (0, 0.5, 0) = (-0.5, 0, 0)
    // Abs Position of attach point = p_WB + (-0.5, 0, 0) = (1.5, 0, 0)
    // Distance to anchor_W (0,0,0) = 1.5 meters.
    // Direction vector (World -> Body) = (1, 0, 0)
    
    // 4. Expected Force Analysis:
    // Extension = 1.5 - rest_length(1.0) = 0.5 m
    // Velocity of attach point:
    // v_point = v_com + w x r = (1,0,0) + 0 = (1,0,0)
    // vel_rel = dot((1,0,0), dir(1,0,0)) = 1.0 m/s
    // Force Mag = -k(0.5) - c(1.0) = -100*0.5 - 10*1 = -50 - 10 = -60 N
    // Force Vector = (-60, 0, 0) (Pulling back towards origin)

    // 5. Expected Torque Analysis:
    // r = (-0.5, 0, 0)
    // F = (-60, 0, 0)
    // tau = r x F = (-0.5 * 0) - (0 * -60) ... = 0 (Force is collinear with radius in this specific setup)
    
    // Let's modify Setup slightly to generate Torque:
    // Move anchor_W to (0, 1, 0).
    // New setup in code below:
    
    Vec3 anchor_W_offset(0.0, 1.0, 0.0);
    LinearSpringDamper spring2(anchor_W_offset, anchor_B, k, c, rest_length);
    
    RigidBodyForces forces;
    spring2.apply(state, forces);

    // Re-calc for spring2:
    // Attach Point (P) = (1.5, 0, 0)
    // Anchor W (A)     = (0.0, 1.0, 0.0)
    // Vector A->P      = (1.5, -1.0, 0.0)
    // Dist             = sqrt(1.5^2 + 1^2) = sqrt(2.25 + 1) = sqrt(3.25) approx 1.80277
    // Dir              = (1.5, -1.0, 0) / 1.80277...
    
    // We check purely if forces are populated non-zero and signs make sense
    // The force should pull P towards A.
    // P is at y=0, A is at y=1. Force y component should be positive.
    REQUIRE(forces.f_W.y() > 0.01); 
    // P is at x=1.5, A is at x=0. Force x component should be negative.
    REQUIRE(forces.f_W.x() < -0.01);
    
    // Torque check:
    // Force acts at P=(1.5,0,0) relative to World Origin, but torque is about COM.
    // COM is at (2,0,0). Lever arm r = P - COM = (-0.5, 0, 0).
    // F has +Y component.
    // r(-x) cross F(+y) => Torque in -Z direction.
    REQUIRE(forces.tau_W.z() < -0.001);
}