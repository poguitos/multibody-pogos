#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>

#include "mbd/joint.hpp"
#include "mbd/system.hpp"
#include "mbd/simulator.hpp"

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
    Mat3 R_90x = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
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

// ============================================================================
// SphericalCoordJoint tests
// ============================================================================

TEST_CASE("SphericalCoordJoint at q=0 gives identity", "[joint][spherical]")
{
    using namespace mbd;

    SphericalCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    REQUIRE(joint.num_dof() == 3);

    VecX q = VecX::Zero(3);
    Transform3 X_J = joint.joint_transform(q);

    require_transform_near(X_J, Transform3::Identity(), eps);
}

TEST_CASE("SphericalCoordJoint rotation about Z matches revolute",
          "[joint][spherical]")
{
    using namespace mbd;

    SphericalCoordJoint sph(Transform3::Identity(), Transform3::Identity(),
                            kGroundIndex, 1);
    RevoluteCoordJoint  rev(Transform3::Identity(), Transform3::Identity(),
                            kGroundIndex, 1);

    // Rotate pi/3 about Z using both joints
    VecX q_sph(3);
    q_sph << 0.0, 0.0, pi / 3.0;

    VecX q_rev(1);
    q_rev << pi / 3.0;

    Transform3 X_sph = sph.joint_transform(q_sph);
    Transform3 X_rev = rev.joint_transform(q_rev);

    require_transform_near(X_sph, X_rev, eps);
}

TEST_CASE("SphericalCoordJoint rotation about arbitrary axis",
          "[joint][spherical]")
{
    using namespace mbd;

    SphericalCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    // Rotate pi/2 about (1,1,0)/sqrt(2)
    const Real angle = pi / 2.0;
    const Vec3 axis = Vec3(1.0, 1.0, 0.0).normalized();
    VecX q(3);
    q = axis * angle;

    Transform3 X_J = joint.joint_transform(q);

    // Reference
    Mat3 R_ref = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    Transform3 X_ref(R_ref, Vec3::Zero());

    require_transform_near(X_J, X_ref, eps);
}

TEST_CASE("SphericalCoordJoint motion subspace at q=0 is identity (top 3 rows)",
          "[joint][spherical]")
{
    using namespace mbd;

    SphericalCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    VecX q = VecX::Zero(3);
    auto S = joint.motion_subspace(q);

    REQUIRE(S.rows() == 6);
    REQUIRE(S.cols() == 3);

    // At q=0, E = I, so S = [I; 0]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(S(i, j), WithinAbs(expected, eps));
        }
    }
    for (int i = 3; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE_THAT(S(i, j), WithinAbs(0.0, eps));
        }
    }
}

TEST_CASE("SphericalCoordJoint in simulator conserves energy",
          "[joint][spherical][energy]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.3, 0.2, 0.1));
    sys.add_body(inertia, RigidBodyState{}, "ball", kGroundIndex);

    // Spherical joint at origin, child COM offset 0.5m along child X
    sys.add_joint(std::make_unique<SphericalCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, 1));

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start tilted, with some angular velocity about Y
    sys.q << 0.0, 0.3, 0.0;
    sys.q_dot << 0.0, 1.0, 0.0;
    sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        sys.compute_kinematics();
        const MatX M = compute_mass_matrix(sys);
        const Real KE = 0.5 * sys.q_dot.transpose() * M * sys.q_dot;
        const auto& st = sys.states[1];
        const Mat3 R = st.q_WB.toRotationMatrix();
        const Vec3 com_W = st.p_WB + R * inertia.com_B;
        return KE + inertia.mass * g_accel * com_W.y();
    };

    const Real E0 = compute_energy();
    sim.run(2.0, 0.001);
    const Real E_final = compute_energy();

    const Real rel_error = std::abs(E_final - E0) / std::abs(E0);
    REQUIRE(rel_error < 1e-4);
}

// ============================================================================
// UniversalCoordJoint tests
// ============================================================================

TEST_CASE("UniversalCoordJoint at q=0 gives identity", "[joint][universal]")
{
    using namespace mbd;

    UniversalCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    REQUIRE(joint.num_dof() == 2);

    VecX q = VecX::Zero(2);
    Transform3 X_J = joint.joint_transform(q);

    require_transform_near(X_J, Transform3::Identity(), eps);
}

TEST_CASE("UniversalCoordJoint single-axis rotations match revolute",
          "[joint][universal]")
{
    using namespace mbd;

    UniversalCoordJoint uni(Transform3::Identity(), Transform3::Identity(),
                            kGroundIndex, 1);

    // q = (theta_z, 0): pure rotation about Z
    {
        VecX q(2);
        q << pi / 4.0, 0.0;
        Transform3 X_J = uni.joint_transform(q);

        Mat3 R_ref = Eigen::AngleAxisd(pi / 4.0, Vec3::UnitZ()).toRotationMatrix();
        Transform3 X_ref(R_ref, Vec3::Zero());

        require_transform_near(X_J, X_ref, eps);
    }

    // q = (0, theta_x): pure rotation about X
    {
        VecX q(2);
        q << 0.0, pi / 3.0;
        Transform3 X_J = uni.joint_transform(q);

        Mat3 R_ref = Eigen::AngleAxisd(pi / 3.0, Vec3::UnitX()).toRotationMatrix();
        Transform3 X_ref(R_ref, Vec3::Zero());

        require_transform_near(X_J, X_ref, eps);
    }
}

TEST_CASE("UniversalCoordJoint combined rotation is Rz * Rx",
          "[joint][universal]")
{
    using namespace mbd;

    UniversalCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    VecX q(2);
    q << 0.5, 0.7;

    Transform3 X_J = joint.joint_transform(q);

    Mat3 Rz = Eigen::AngleAxisd(0.5, Vec3::UnitZ()).toRotationMatrix();
    Mat3 Rx = Eigen::AngleAxisd(0.7, Vec3::UnitX()).toRotationMatrix();
    Transform3 X_ref(Rz * Rx, Vec3::Zero());

    require_transform_near(X_J, X_ref, eps);
}

TEST_CASE("UniversalCoordJoint motion subspace at q=0",
          "[joint][universal]")
{
    using namespace mbd;

    UniversalCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    VecX q = VecX::Zero(2);
    auto S = joint.motion_subspace(q);

    REQUIRE(S.rows() == 6);
    REQUIRE(S.cols() == 2);

    // Column 0: [0,0,1, 0,0,0] (Z rotation)
    REQUIRE_THAT(S(0, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(1, 0), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(2, 0), WithinAbs(1.0, eps));

    // Column 1: [1,0,0, 0,0,0] (X rotation, since Rz(0) = I)
    REQUIRE_THAT(S(0, 1), WithinAbs(1.0, eps));
    REQUIRE_THAT(S(1, 1), WithinAbs(0.0, eps));
    REQUIRE_THAT(S(2, 1), WithinAbs(0.0, eps));

    // Linear rows zero
    for (int j = 0; j < 2; ++j) {
        REQUIRE_THAT(S(3, j), WithinAbs(0.0, eps));
        REQUIRE_THAT(S(4, j), WithinAbs(0.0, eps));
        REQUIRE_THAT(S(5, j), WithinAbs(0.0, eps));
    }
}

// ============================================================================
// FreeCoordJoint tests
// ============================================================================

TEST_CASE("FreeCoordJoint at q=0 gives identity", "[joint][free]")
{
    using namespace mbd;

    FreeCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                         kGroundIndex, 1);

    REQUIRE(joint.num_dof() == 6);

    VecX q = VecX::Zero(6);
    Transform3 X_J = joint.joint_transform(q);

    require_transform_near(X_J, Transform3::Identity(), eps);
}

TEST_CASE("FreeCoordJoint pure translation", "[joint][free]")
{
    using namespace mbd;

    FreeCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                         kGroundIndex, 1);

    VecX q = VecX::Zero(6);
    q(0) = 1.0;
    q(1) = 2.0;
    q(2) = 3.0;

    Transform3 X_J = joint.joint_transform(q);

    Vec3 origin = X_J * Vec3::Zero();
    require_vec3_near(origin, Vec3(1.0, 2.0, 3.0), eps);

    // No rotation
    Vec3 x_axis = X_J.rotate(Vec3::UnitX());
    require_vec3_near(x_axis, Vec3::UnitX(), eps);
}

TEST_CASE("FreeCoordJoint pure rotation matches spherical", "[joint][free]")
{
    using namespace mbd;

    FreeCoordJoint free_j(Transform3::Identity(), Transform3::Identity(),
                          kGroundIndex, 1);
    SphericalCoordJoint sph_j(Transform3::Identity(), Transform3::Identity(),
                              kGroundIndex, 1);

    VecX q_free(6);
    q_free << 0.0, 0.0, 0.0, 0.3, -0.5, 0.7;

    VecX q_sph(3);
    q_sph << 0.3, -0.5, 0.7;

    Transform3 X_free = free_j.joint_transform(q_free);
    Transform3 X_sph  = sph_j.joint_transform(q_sph);

    require_transform_near(X_free, X_sph, eps);
}

TEST_CASE("FreeCoordJoint motion subspace at q=0 is 6x6 identity-like",
          "[joint][free]")
{
    using namespace mbd;

    FreeCoordJoint joint(Transform3::Identity(), Transform3::Identity(),
                         kGroundIndex, 1);

    VecX q = VecX::Zero(6);
    auto S = joint.motion_subspace(q);

    REQUIRE(S.rows() == 6);
    REQUIRE(S.cols() == 6);

    // At q=0: columns 0-2 map to linear velocity (rows 3-5)
    //         columns 3-5 map to angular velocity (rows 0-2) via E=I
    Eigen::Matrix<Real, 6, 6> S_expected;
    S_expected.setZero();
    S_expected.block<3,3>(3, 0) = Mat3::Identity(); // linear
    S_expected.block<3,3>(0, 3) = Mat3::Identity(); // angular

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            REQUIRE_THAT(S(i, j), WithinAbs(S_expected(i, j), eps));
        }
    }
}

TEST_CASE("FreeCoordJoint floating body in free fall", "[joint][free][dynamics]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(2.0, Vec3(0.3, 0.2, 0.1));
    sys.add_body(inertia, RigidBodyState{}, "chassis", kGroundIndex);

    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at (0, 10, 0), no rotation, no velocity
    sys.q << 0.0, 10.0, 0.0, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    const Real T = 1.0;
    sim.run(T, 0.001);

    // Translation: free fall along Y
    const Real y_expected = 10.0 - 0.5 * g_accel * T * T;
    const Real vy_expected = -g_accel * T;

    REQUIRE_THAT(sys.q(0), WithinAbs(0.0, 1e-6));         // x unchanged
    REQUIRE_THAT(sys.q(1), WithinAbs(y_expected, 1e-5));   // y = free fall
    REQUIRE_THAT(sys.q(2), WithinAbs(0.0, 1e-6));         // z unchanged

    REQUIRE_THAT(sys.q_dot(1), WithinAbs(vy_expected, 1e-5));

    // No rotation
    REQUIRE_THAT(sys.q(3), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(sys.q(4), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(sys.q(5), WithinAbs(0.0, 1e-6));

    // World position matches
    require_vec3_near(sys.states[1].p_WB, Vec3(0.0, y_expected, 0.0), 1e-5);
}

TEST_CASE("FreeCoordJoint spinning body conserves angular momentum",
          "[joint][free][dynamics]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.3, 0.2, 0.1));
    sys.add_body(inertia, RigidBodyState{}, "spinner", kGroundIndex);

    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero()); // No gravity — pure rotation
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start spinning about X at 5 rad/s
    sys.q.setZero();
    sys.q_dot << 0.0, 0.0, 0.0, 5.0, 0.0, 0.0;
    sys.compute_kinematics();

    // Compute initial angular momentum
    const Mat3 R0 = sys.states[1].q_WB.toRotationMatrix();
    const Mat3 I_W0 = R0 * inertia.I_com_B * R0.transpose();
    const Vec3 L0 = I_W0 * sys.states[1].w_WB;

    sim.run(2.0, 0.001);

    // Compute final angular momentum
    const Mat3 R1 = sys.states[1].q_WB.toRotationMatrix();
    const Mat3 I_W1 = R1 * inertia.I_com_B * R1.transpose();
    const Vec3 L1 = I_W1 * sys.states[1].w_WB;

    require_vec3_near(L1, L0, 1e-4);
}