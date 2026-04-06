#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>

#include "mbd/system.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-10;

    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }
}

// ============================================================================
// Pose FK
// ============================================================================

TEST_CASE("FK: single revolute pendulum pose", "[fk]")
{
    using namespace mbd;

    MultibodySystem sys;

    // Link: 1m long bar, mass 1kg
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);

    // Revolute at origin, axis Z. Joint frame at parent origin.
    // Child joint frame at (-0.5, 0, 0) in child body = left end of link.
    auto joint = std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, b1);

    sys.add_joint(std::move(joint));

    REQUIRE(sys.total_dof == 1);
    REQUIRE(sys.q.size() == 1);
    REQUIRE(sys.q_dot.size() == 1);

    // q = 0: link extends along +X, COM at (0.5, 0, 0)
    sys.q(0) = 0.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.5, 0.0, 0.0), eps);

    // q = pi/2: link extends along +Y, COM at (0, 0.5, 0)
    sys.q(0) = pi / 2.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.0, 0.5, 0.0), eps);

    // q = pi: link extends along -X, COM at (-0.5, 0, 0)
    sys.q(0) = pi;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(-0.5, 0.0, 0.0), eps);
}

TEST_CASE("FK: double revolute pendulum pose", "[fk]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));

    // Link 1: COM at center of 1m bar
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);
    // Link 2
    BodyIndex b2 = sys.add_body(inertia, RigidBodyState{}, "link2", b1);

    // Joint 1: ground to link1. Joint at origin.
    // Child joint frame at left end of link (-0.5, 0, 0)
    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, b1));

    // Joint 2: link1 to link2. Joint at right end of link1 (+0.5, 0, 0) in link1 frame.
    // Child joint frame at left end of link2 (-0.5, 0, 0)
    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0)),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        b1, b2));

    REQUIRE(sys.total_dof == 2);

    // Both at zero: link1 COM at (0.5,0,0), link2 COM at (1.5,0,0)
    sys.q << 0.0, 0.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.5, 0.0, 0.0), eps);
    require_vec3_near(sys.states[b2].p_WB, Vec3(1.5, 0.0, 0.0), eps);

    // Joint1 = pi/2, joint2 = 0: both links form an L going up
    // Link1 COM at (0, 0.5, 0), link1 tip at (0, 1, 0)
    // Link2 extends along link1's rotated X = world Y
    // Link2 COM at (0, 1.5, 0)
    sys.q << pi / 2.0, 0.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.0, 0.5, 0.0), eps);
    require_vec3_near(sys.states[b2].p_WB, Vec3(0.0, 1.5, 0.0), eps);

    // Joint1 = pi/2, joint2 = -pi/2: link2 folds back horizontal
    // Link1 tip at (0, 1, 0). Joint2 rotates link2 by -90 in link1's frame.
    // Link1's X-axis is world +Y. Joint2 at -90 rotates link2's local X
    // back towards world +X.
    // Link2 COM = joint2_pos + R_W2 * (0.5, 0, 0)
    // R_W2 = Rz(pi/2) * Rz(-pi/2) = I, so link2's X-axis = world X
    // link2 COM = (0, 1, 0) + (0.5, 0, 0) = (0.5, 1.0, 0)
    sys.q << pi / 2.0, -pi / 2.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b2].p_WB, Vec3(0.5, 1.0, 0.0), eps);
}

TEST_CASE("FK: prismatic joint translates child", "[fk]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "slider", kGroundIndex);

    // Prismatic along Z (default joint axis)
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3::Identity(),
        Transform3::Identity(),
        kGroundIndex, b1));

    sys.q(0) = 3.5;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.0, 0.0, 3.5), eps);

    sys.q(0) = -1.2;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.0, 0.0, -1.2), eps);
}

TEST_CASE("FK: prismatic along X via rotated joint frame", "[fk]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "slider_x", kGroundIndex);

    // Rotate joint frame so Z aligns with parent X: Ry(+pi/2) maps Z -> +X
    Mat3 R = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitY()).toRotationMatrix();
    Transform3 X_J_frame = Transform3::FromRotation(R);

    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_J_frame, X_J_frame,
        kGroundIndex, b1));

    sys.q(0) = 4.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(4.0, 0.0, 0.0), eps);
}

TEST_CASE("FK: fixed joint preserves offset", "[fk]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "fixed_child", kGroundIndex);

    // Fixed joint: child is at (1, 2, 3) relative to parent
    sys.add_joint(std::make_unique<FixedJoint>(
        Transform3::FromTranslation(Vec3(1.0, 2.0, 3.0)),
        Transform3::Identity(),
        kGroundIndex, b1));

    REQUIRE(sys.total_dof == 0);

    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(1.0, 2.0, 3.0), eps);
}

TEST_CASE("FK: mixed chain revolute + prismatic", "[fk]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));

    // Body 1: rotating arm, 1m long
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "arm", kGroundIndex);
    // Body 2: slider on the arm
    BodyIndex b2 = sys.add_body(inertia, RigidBodyState{}, "slider", b1);

    // Joint 1: revolute at origin, Z-axis
    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::Identity(),
        kGroundIndex, b1));

    // Joint 2: prismatic along arm's local Z
    // But we want it along the arm's X-axis, so rotate joint frame:
    // Ry(+pi/2) maps joint Z to parent X
    Mat3 R_y90 = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitY()).toRotationMatrix();
    Transform3 X_slide = Transform3::FromRotation(R_y90);

    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_slide, X_slide,
        b1, b2));

    REQUIRE(sys.total_dof == 2);

    // Arm at 0 deg, slider at 2m along arm X (= world X)
    sys.q << 0.0, 2.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b1].p_WB, Vec3(0.0, 0.0, 0.0), eps);
    require_vec3_near(sys.states[b2].p_WB, Vec3(2.0, 0.0, 0.0), eps);

    // Arm at 90 deg, slider at 2m along arm X (= world Y now)
    sys.q << pi / 2.0, 2.0;
    sys.compute_forward_kinematics();

    require_vec3_near(sys.states[b2].p_WB, Vec3(0.0, 2.0, 0.0), eps);
}

// ============================================================================
// Velocity FK
// ============================================================================

TEST_CASE("Velocity FK: single revolute pendulum", "[fk][velocity]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, b1));

    // Pendulum horizontal (q=0), spinning at 2 rad/s
    sys.q(0) = 0.0;
    sys.q_dot(0) = 2.0;
    sys.compute_kinematics();

    // Angular velocity: [0, 0, 2] (rotation about world Z)
    require_vec3_near(sys.states[b1].w_WB, Vec3(0.0, 0.0, 2.0), eps);

    // Linear velocity of COM at (0.5, 0, 0):
    // v = w x r = (0,0,2) x (0.5,0,0) = (0, 1.0, 0)
    require_vec3_near(sys.states[b1].v_WB, Vec3(0.0, 1.0, 0.0), eps);
}

TEST_CASE("Velocity FK: pendulum at 90 degrees", "[fk][velocity]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, b1));

    // Pendulum vertical (q=pi/2), spinning at 3 rad/s
    sys.q(0) = pi / 2.0;
    sys.q_dot(0) = 3.0;
    sys.compute_kinematics();

    // Angular velocity still about world Z
    require_vec3_near(sys.states[b1].w_WB, Vec3(0.0, 0.0, 3.0), eps);

    // COM at (0, 0.5, 0). v = w x r = (0,0,3) x (0,0.5,0) = (-1.5, 0, 0)
    require_vec3_near(sys.states[b1].v_WB, Vec3(-1.5, 0.0, 0.0), eps);
}

TEST_CASE("Velocity FK: prismatic joint", "[fk][velocity]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "slider", kGroundIndex);

    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3::Identity(),
        Transform3::Identity(),
        kGroundIndex, b1));

    sys.q(0) = 1.0;
    sys.q_dot(0) = 5.0;
    sys.compute_kinematics();

    // No angular velocity
    require_vec3_near(sys.states[b1].w_WB, Vec3::Zero(), eps);

    // Linear velocity along Z at 5 m/s
    require_vec3_near(sys.states[b1].v_WB, Vec3(0.0, 0.0, 5.0), eps);
}

TEST_CASE("Velocity FK: double pendulum", "[fk][velocity]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));

    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);
    BodyIndex b2 = sys.add_body(inertia, RigidBodyState{}, "link2", b1);

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, b1));

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0)),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        b1, b2));

    // Both horizontal (q=0,0). Link1 spinning at 1 rad/s, link2 at 2 rad/s
    sys.q << 0.0, 0.0;
    sys.q_dot << 1.0, 2.0;
    sys.compute_kinematics();

    // Link1: w = (0,0,1), COM at (0.5,0,0), v = (0,0,1)x(0.5,0,0) = (0,0.5,0)
    require_vec3_near(sys.states[b1].w_WB, Vec3(0.0, 0.0, 1.0), eps);
    require_vec3_near(sys.states[b1].v_WB, Vec3(0.0, 0.5, 0.0), eps);

    // Link2: w = (0,0,1+2) = (0,0,3)
    require_vec3_near(sys.states[b2].w_WB, Vec3(0.0, 0.0, 3.0), eps);

    // Link2 COM at (1.5, 0, 0).
    // v_link2 = v_parent + w_parent x r_PC + v_rel
    // v_parent (link1 COM) = (0, 0.5, 0)
    // r_PC = (1.5,0,0) - (0.5,0,0) = (1.0, 0, 0)
    // w_parent x r_PC = (0,0,1) x (1,0,0) = (0, 1, 0)
    // v_rel from joint2: joint2 is at tip of link1 (1,0,0).
    //   w_rel = (0,0,2). The joint frame = world frame (q=0, no rotation).
    //   v_rel_linear = 0 (revolute). But the velocity from the joint
    //   must account for the offset from joint to child COM:
    //   Actually v_rel_W is just the linear part of S*q_dot rotated to world = 0 for revolute.
    //   The (0,0,2) x (child COM - joint pos) is handled by w_child x r.
    //   Let's just verify the total:
    // v_link2_COM = v_link1_COM + w_link1 x r_12 + v_rel_W
    //            = (0, 0.5, 0) + (0, 1.0, 0) + (0, 0, 0) = (0, 1.5, 0)
    // But wait: the child velocity formula uses parent origin, not parent COM.
    // Actually in our formulation, v_WB is the velocity of the body origin (= COM for now).
    // r_PC = child_origin - parent_origin.
    //
    // Let me just compute expected value directly:
    // Joint2 position = (1, 0, 0). Its velocity = w_link1 x (joint2_pos - link1_origin)
    //   But link1 origin = link1 COM = (0.5, 0, 0). Joint2 is at (1,0,0) in world.
    //   v_joint2 = v_link1_COM + w_link1 x ((1,0,0) - (0.5,0,0))
    //            = (0, 0.5, 0) + (0,0,1) x (0.5,0,0) = (0, 0.5, 0) + (0, 0.5, 0) = (0, 1.0, 0)
    // Link2 COM = (1.5, 0, 0), offset from joint2 = (0.5, 0, 0)
    // v_link2_COM = v_joint2 + w_link2 x (0.5, 0, 0)
    //            = (0, 1.0, 0) + (0,0,3) x (0.5,0,0) = (0, 1.0, 0) + (0, 1.5, 0) = (0, 2.5, 0)
    //
    // But our formula computes: v_child = v_parent + w_parent x r_PC + v_rel_W
    // v_parent = v_link1 = (0, 0.5, 0)  (velocity of link1 origin = link1 COM)
    // r_PC = link2_COM - link1_COM = (1.0, 0, 0)
    // w_parent x r_PC = (0,0,1) x (1,0,0) = (0, 1, 0)
    // v_rel_W = 0 (revolute)
    // Total = (0, 0.5, 0) + (0, 1, 0) + 0 = (0, 1.5, 0)
    //
    // This misses the contribution of w_link2 acting on the offset from joint to link2 COM.
    // Expected correct answer is (0, 2.5, 0). So (0, 1.5, 0) is what our formula gives.
    // The discrepancy is because v_WB is the velocity of the body origin (COM), and
    // the w x r formula only captures the parent's contribution, not the child's own
    // angular velocity acting on the child's offset from the joint.
    //
    // The correct recursive formula should be:
    //   v_child_joint = v_parent + w_parent x r_parent_to_joint
    //   v_child_origin = v_child_joint + w_child x r_joint_to_child_origin
    //
    // This is a known subtlety. Our current formula in compute_forward_velocities
    // uses r_PC (parent origin to child origin) which bundles both offsets.
    // With only parent w, it misses the child w acting on the joint-to-child offset.
    // We need to fix compute_forward_velocities.
    //
    // For now, let's just check what we expect with the corrected formula:
    require_vec3_near(sys.states[b2].v_WB, Vec3(0.0, 2.5, 0.0), eps);
}

TEST_CASE("Velocity FK: zero q_dot gives zero velocities", "[fk][velocity]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));

    BodyIndex b1 = sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);
    BodyIndex b2 = sys.add_body(inertia, RigidBodyState{}, "link2", b1);

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, b1));

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0)),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        b1, b2));

    sys.q << 0.7, -0.3;
    sys.q_dot << 0.0, 0.0;
    sys.compute_kinematics();

    require_vec3_near(sys.states[b1].v_WB, Vec3::Zero(), eps);
    require_vec3_near(sys.states[b1].w_WB, Vec3::Zero(), eps);
    require_vec3_near(sys.states[b2].v_WB, Vec3::Zero(), eps);
    require_vec3_near(sys.states[b2].w_WB, Vec3::Zero(), eps);
}