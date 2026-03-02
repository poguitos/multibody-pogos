#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>

#include "mbd/joint.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-12;

    // Helper: check two Vec3 are equal within tolerance
    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }

    // Helper: check two Transform3 produce same result on a test point
    void require_transform_near(const mbd::Transform3& A,
                                const mbd::Transform3& B, double tol)
    {
        mbd::Vec3 test_pt(0.7, -1.3, 2.1);
        require_vec3_near(A * test_pt, B * test_pt, tol);
    }
}

// ============================================================================
// RevoluteCoordJoint tests
// ============================================================================

TEST_CASE("RevoluteCoordJoint at q=0 gives identity joint transform",
          "[joint][revolute]")
{
    using namespace mbd;

    RevoluteCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                             kGroundIndex, 1);

    REQUIRE(joint.num_dof() == 1);

    VecX q(1);
    q << 0.0;

    Transform3 X_J = joint.joint_transform(q);

    // Should be identity
    require_transform_near(X_J, Transform3::Identity(), eps);
}

TEST_CASE("RevoluteCoordJoint at q=pi/2 rotates 90 degrees about Z",
          "[joint][revolute]")
{
    using namespace mbd;

    RevoluteCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                             kGroundIndex, 1);

    VecX q(1);
    q << pi / 2.0;

    Transform3 X_J = joint.joint_transform(q);

    // X-axis should map to Y-axis
    Vec3 x_in(1.0, 0.0, 0.0);
    Vec3 x_out = X_J.rotate(x_in);

    require_vec3_near(x_out, Vec3(0.0, 1.0, 0.0), eps);

    // Z-axis unchanged
    Vec3 z_in(0.0, 0.0, 1.0);
    Vec3 z_out = X_J.rotate(z_in);

    require_vec3_near(z_out, Vec3(0.0, 0.0, 1.0), eps);
}

TEST_CASE("RevoluteCoordJoint motion subspace is constant [0,0,1,0,0,0]^T",
          "[joint][revolute]")
{
    using namespace mbd;

    RevoluteCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                             kGroundIndex, 1);

    VecX q(1);
    q << 1.23; // arbitrary angle

    auto S = joint.motion_subspace(q);

    REQUIRE(S.rows() == 6);
    REQUIRE(S.cols() == 1);

    REQUIRE_THAT(S(0, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(1, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(2, 0), WithinAbs(1.0, eps));
    REQUIRE_THAT(S(3, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(4, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(5, 0), WithinAbs(0.0, eps));
}

TEST_CASE("RevoluteCoordJoint parent_to_child_transform with non-trivial frames",
          "[joint][revolute]")
{
    using namespace mbd;

    // Joint frame offset 1m along parent X
    Transform3 X_PJ = Transform3::FromTranslation(Vec3(1.0, 0.0, 0.0));
    // Joint frame offset 0.5m along child X
    Transform3 X_CJ = Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0));

    RevoluteCoordJoint joint(X_PJ, X_CJ, kGroundIndex, 1);

    VecX q(1);
    q << 0.0;

    Transform3 X_PC = joint.parent_to_child_transform(q);

    // At q=0, X_PC = X_PJ * I * X_CJ^-1
    // X_PJ translates +1 along X, X_CJ^-1 translates -0.5 along X
    // Net: translation of +0.5 along X
    Vec3 origin_child_in_parent = X_PC * Vec3::Zero();
    require_vec3_near(origin_child_in_parent, Vec3(0.5, 0.0, 0.0), eps);

    // Now rotate 90 degrees
    q << pi / 2.0;
    X_PC = joint.parent_to_child_transform(q);

    // X_PC = X_PJ * Rz(90) * X_CJ^-1
    // X_CJ^-1 translates (-0.5, 0, 0)
    // Rz(90) * (-0.5, 0, 0) = (0, -0.5, 0)
    // X_PJ adds (1, 0, 0)
    // Total origin: (1.0, -0.5, 0.0)
    origin_child_in_parent = X_PC * Vec3::Zero();
    require_vec3_near(origin_child_in_parent, Vec3(1.0, -0.5, 0.0), eps);
}

TEST_CASE("RevoluteCoordJoint composition: double pendulum kinematics",
          "[joint][revolute]")
{
    using namespace mbd;

    // Joint 1: ground to link1, joint at origin, axis Z
    // Link1 extends 1m along X (joint at its left end)
    Transform3 X_PJ1 = Transform3::Identity();
    Transform3 X_CJ1 = Transform3::Identity();
    RevoluteCoordJoint joint1(X_PJ1, X_CJ1, kGroundIndex, 1);

    // Joint 2: link1 to link2, joint at (1,0,0) in link1 frame
    Transform3 X_PJ2 = Transform3::FromTranslation(Vec3(1.0, 0.0, 0.0));
    Transform3 X_CJ2 = Transform3::Identity();
    RevoluteCoordJoint joint2(X_PJ2, X_CJ2, 1, 2);

    // Both joints at 0: link2 origin at (1,0,0)
    VecX q0(1);
    q0 << 0.0;

    Transform3 X_W1 = joint1.parent_to_child_transform(q0);
    Transform3 X_W2 = X_W1 * joint2.parent_to_child_transform(q0);

    require_vec3_near(X_W2 * Vec3::Zero(), Vec3(1.0, 0.0, 0.0), eps);

    // Joint1 at 90 deg, joint2 at 0: link2 origin at (0, 1, 0)
    VecX q90(1);
    q90 << pi / 2.0;

    X_W1 = joint1.parent_to_child_transform(q90);
    X_W2 = X_W1 * joint2.parent_to_child_transform(q0);

    require_vec3_near(X_W2 * Vec3::Zero(), Vec3(0.0, 1.0, 0.0), eps);

    // Joint1 at 90, joint2 at 90: link2 origin at (-1, 1, 0)
    // Link1 tip at (0,1,0). Link2 rotated another 90 from link1's frame:
    // link1's X-axis points in world Y direction at q1=90,
    // so link2 rotated 90 from that points in world -X.
    // But X_PJ2 translation (1,0,0) in link1 frame maps to (0,1,0) in world.
    // Then joint2 rotation makes link2's local X point in world -X.
    // link2 origin = link1 origin + R_W1*(1,0,0) = (0,0,0) + (0,1,0) = (0,1,0)
    // That's the joint location. Link2 origin in its own frame is (0,0,0).
    X_W2 = X_W1 * joint2.parent_to_child_transform(q90);

    require_vec3_near(X_W2 * Vec3::Zero(), Vec3(0.0, 1.0, 0.0), eps);

    // Tip of link2 (1m along its local X):
    // R_W2 rotates local X to world -X direction
    // So tip at (0,1,0) + (-1, 0, 0) = (-1, 1, 0)
    require_vec3_near(X_W2 * Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 1.0, 0.0), eps);
}

// ============================================================================
// PrismaticCoordJoint tests
// ============================================================================

TEST_CASE("PrismaticCoordJoint at q=0 gives identity joint transform",
          "[joint][prismatic]")
{
    using namespace mbd;

    PrismaticCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    REQUIRE(joint.num_dof() == 1);

    VecX q(1);
    q << 0.0;

    Transform3 X_J = joint.joint_transform(q);
    require_transform_near(X_J, Transform3::Identity(), eps);
}

TEST_CASE("PrismaticCoordJoint translates along Z by q",
          "[joint][prismatic]")
{
    using namespace mbd;

    PrismaticCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    VecX q(1);
    q << 2.5;

    Transform3 X_J = joint.joint_transform(q);

    Vec3 origin = X_J * Vec3::Zero();
    require_vec3_near(origin, Vec3(0.0, 0.0, 2.5), eps);

    // No rotation
    Vec3 x_axis = X_J.rotate(Vec3::UnitX());
    require_vec3_near(x_axis, Vec3::UnitX(), eps);
}

TEST_CASE("PrismaticCoordJoint motion subspace is [0,0,0,0,0,1]^T",
          "[joint][prismatic]")
{
    using namespace mbd;

    PrismaticCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    VecX q(1);
    q << 0.0;

    auto S = joint.motion_subspace(q);

    REQUIRE(S.rows() == 6);
    REQUIRE(S.cols() == 1);

    REQUIRE_THAT(S(0, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(1, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(2, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(3, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(4, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(5, 0), WithinAbs(1.0, eps));
}

TEST_CASE("PrismaticCoordJoint with rotated joint frame slides along world Y",
          "[joint][prismatic]")
{
    using namespace mbd;

    // Rotate joint frame so that joint Z aligns with parent Y
    // Rotation: 90 deg about X maps Z -> Y
    // Actually Rx(90): Y->-Z, Z->Y. So joint Z maps to parent Y. Correct.
    Mat3 R_90x = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    Transform3 X_PJ = Transform3::FromRotation(R_90x);
    Transform3 X_CJ = Transform3::FromRotation(R_90x);

    PrismaticCoordJoint joint(X_PJ, X_CJ, kGroundIndex, 1);

    VecX q(1);
    q << 3.0;

    Transform3 X_PC = joint.parent_to_child_transform(q);

    // Joint slides 3m along joint Z, which is parent Y
    Vec3 origin = X_PC * Vec3::Zero();
    require_vec3_near(origin, Vec3(0.0, 3.0, 0.0), eps);
}

// ============================================================================
// FixedJoint tests
// ============================================================================

TEST_CASE("FixedJoint has 0 DOF and identity joint transform", "[joint][fixed]")
{
    using namespace mbd;

    Transform3 X_PJ = Transform3::FromTranslation(Vec3(1.0, 2.0, 3.0));
    Transform3 X_CJ = Transform3::FromTranslation(Vec3(0.1, 0.2, 0.3));

    FixedJoint joint(X_PJ, X_CJ, kGroundIndex, 1);

    REQUIRE(joint.num_dof() == 0);

    VecX q_empty(0);
    Transform3 X_J = joint.joint_transform(q_empty);
    require_transform_near(X_J, Transform3::Identity(), eps);

    auto S = joint.motion_subspace(q_empty);
    REQUIRE(S.rows() == 6);
    REQUIRE(S.cols() == 0);
}

TEST_CASE("FixedJoint parent_to_child_transform is X_PJ * X_CJ_inv",
          "[joint][fixed]")
{
    using namespace mbd;

    Vec3 t_pj(1.0, 0.0, 0.0);
    Vec3 t_cj(0.0, 0.5, 0.0);

    FixedJoint joint(Transform3::FromTranslation(t_pj),
                     Transform3::FromTranslation(t_cj),
                     kGroundIndex, 1);

    VecX q_empty(0);
    Transform3 X_PC = joint.parent_to_child_transform(q_empty);

    // X_PC = T(1,0,0) * I * T(0,0.5,0)^-1 = T(1,0,0) * T(0,-0.5,0) = T(1,-0.5,0)
    Vec3 origin = X_PC * Vec3::Zero();
    require_vec3_near(origin, Vec3(1.0, -0.5, 0.0), eps);
}

// ============================================================================
// Bias acceleration
// ============================================================================

TEST_CASE("Simple joints have zero bias acceleration", "[joint]")
{
    using namespace mbd;

    RevoluteCoordJoint rev(Transform3::Identity(), Transform3::Identity(),
                           kGroundIndex, 1);
    PrismaticCoordJoint pri(Transform3::Identity(), Transform3::Identity(),
                            kGroundIndex, 1);

    VecX q(1);
    q << 1.5;
    VecX qd(1);
    qd << 2.0;

    Vec6 bias_rev = rev.bias_acceleration(q, qd);
    Vec6 bias_pri = pri.bias_acceleration(q, qd);

    for (int i = 0; i < 6; ++i) {
        REQUIRE_THAT(bias_rev(i), WithinAbs(0.0, eps));
        REQUIRE_THAT(bias_pri(i), WithinAbs(0.0, eps));
    }
}