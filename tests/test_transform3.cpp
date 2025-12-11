#include <catch2/catch_all.hpp>
#include <Eigen/Geometry>

#include "mbd/math.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-12;
}

TEST_CASE("Transform3 identity behaves as expected", "[math][transform3]")
{
    using namespace mbd;

    const Transform3 I_default;
    const Transform3 I_static = Transform3::Identity();

    // A generic test point
    const Vec3 x_local(1.2, -3.4, 5.6);

    // Identity should not change the point
    const Vec3 x_world_1 = I_default * x_local;
    const Vec3 x_world_2 = I_static * x_local;

    REQUIRE_THAT(x_world_1.x(), WithinAbs(x_local.x(), eps));
    REQUIRE_THAT(x_world_1.y(), WithinAbs(x_local.y(), eps));
    REQUIRE_THAT(x_world_1.z(), WithinAbs(x_local.z(), eps));

    REQUIRE_THAT(x_world_2.x(), WithinAbs(x_local.x(), eps));
    REQUIRE_THAT(x_world_2.y(), WithinAbs(x_local.y(), eps));
    REQUIRE_THAT(x_world_2.z(), WithinAbs(x_local.z(), eps));

    // Check that translation is zero
    REQUIRE_THAT(I_default.p.x(), WithinAbs(0.0, eps));
    REQUIRE_THAT(I_default.p.y(), WithinAbs(0.0, eps));
    REQUIRE_THAT(I_default.p.z(), WithinAbs(0.0, eps));

    // Check that rotation is identity by applying it
    const Vec3 Rv = I_default.R * x_local;
    REQUIRE_THAT(Rv.x(), WithinAbs(x_local.x(), eps));
    REQUIRE_THAT(Rv.y(), WithinAbs(x_local.y(), eps));
    REQUIRE_THAT(Rv.z(), WithinAbs(x_local.z(), eps));
}

TEST_CASE("Transform3 composition is consistent with sequential application",
          "[math][transform3]")
{
    using namespace mbd;

    // Build two simple rotations + translations
    const Real angle1 = 0.3;  // rad
    const Real angle2 = -0.5; // rad

    Mat3 R1 = Eigen::AngleAxisd(angle1, Vec3::UnitZ()).toRotationMatrix();
    Mat3 R2 = Eigen::AngleAxisd(angle2, Vec3::UnitX()).toRotationMatrix();

    Vec3 p1(1.0, 0.5, -0.2);
    Vec3 p2(-0.3, 1.2, 0.7);

    Transform3 T1(R1, p1);
    Transform3 T2(R2, p2);

    Transform3 T12 = T1 * T2;

    // Test point
    const Vec3 x_local(0.7, -1.1, 2.3);

    const Vec3 x_world_seq  = T1 * (T2 * x_local);
    const Vec3 x_world_comb = T12 * x_local;

    REQUIRE_THAT(x_world_seq.x(), WithinAbs(x_world_comb.x(), eps));
    REQUIRE_THAT(x_world_seq.y(), WithinAbs(x_world_comb.y(), eps));
    REQUIRE_THAT(x_world_seq.z(), WithinAbs(x_world_comb.z(), eps));
}

TEST_CASE("Transform3 inversion correctly undoes a transform",
          "[math][transform3]")
{
    using namespace mbd;

    const Real angle = 0.7;
    Mat3 R = Eigen::AngleAxisd(angle, Vec3::UnitY()).toRotationMatrix();
    Vec3 p(0.4, -0.9, 1.5);

    Transform3 T(R, p);
    Transform3 T_inv = T.inverse();

    // A couple of test points
    const Vec3 x_local_1(0.0, 0.0, 0.0);
    const Vec3 x_local_2(1.0, 2.0, 3.0);

    const Vec3 x_world_1 = T * x_local_1;
    const Vec3 x_world_2 = T * x_local_2;

    const Vec3 x_back_1 = T_inv * x_world_1;
    const Vec3 x_back_2 = T_inv * x_world_2;

    REQUIRE_THAT(x_back_1.x(), WithinAbs(x_local_1.x(), eps));
    REQUIRE_THAT(x_back_1.y(), WithinAbs(x_local_1.y(), eps));
    REQUIRE_THAT(x_back_1.z(), WithinAbs(x_local_1.z(), eps));

    REQUIRE_THAT(x_back_2.x(), WithinAbs(x_local_2.x(), eps));
    REQUIRE_THAT(x_back_2.y(), WithinAbs(x_local_2.y(), eps));
    REQUIRE_THAT(x_back_2.z(), WithinAbs(x_local_2.z(), eps));

    // Also check T * T^{-1} behaves like identity on a point in world frame
    const Vec3 x_world_test(4.0, -1.0, 0.5);
    const Vec3 x_local_test = T_inv * x_world_test;
    const Vec3 x_world_back = T * x_local_test;

    REQUIRE_THAT(x_world_back.x(), WithinAbs(x_world_test.x(), eps));
    REQUIRE_THAT(x_world_back.y(), WithinAbs(x_world_test.y(), eps));
    REQUIRE_THAT(x_world_back.z(), WithinAbs(x_world_test.z(), eps));
}

TEST_CASE("Transform3 can be built from quaternion and translation",
          "[math][transform3]")
{
    using namespace mbd;

    const Real angle = 0.25;
    const Vec3 axis  = Vec3::UnitZ();

    const Eigen::AngleAxisd aa(angle, axis);
    const Quat q_from_aa(aa);

    const Vec3 p(0.1, -0.2, 0.3);

    // Reference transform built from rotation matrix
    Transform3 T_mat(aa.toRotationMatrix(), p);

    // Construct from quaternion:
    Transform3 T_q_ctor(q_from_aa, p);
    Transform3 T_q_factory = Transform3::FromQuatTranslation(q_from_aa, p);

    const Vec3 x_local(0.4, 0.5, -0.1);

    const Vec3 x_mat = T_mat * x_local;
    const Vec3 x_q1  = T_q_ctor * x_local;
    const Vec3 x_q2  = T_q_factory * x_local;

    REQUIRE_THAT(x_q1.x(), WithinAbs(x_mat.x(), eps));
    REQUIRE_THAT(x_q1.y(), WithinAbs(x_mat.y(), eps));
    REQUIRE_THAT(x_q1.z(), WithinAbs(x_mat.z(), eps));

    REQUIRE_THAT(x_q2.x(), WithinAbs(x_mat.x(), eps));
    REQUIRE_THAT(x_q2.y(), WithinAbs(x_mat.y(), eps));
    REQUIRE_THAT(x_q2.z(), WithinAbs(x_mat.z(), eps));
}

TEST_CASE("Transform3 factory helpers and accessors behave as expected",
          "[math][transform3]")
{
    using namespace mbd;

    // Pure translation
    Vec3 t(1.0, -2.0, 0.5);
    Transform3 T_trans = Transform3::FromTranslation(t);

    const Vec3 x_local(0.5, 0.5, 0.5);
    const Vec3 x_world = T_trans * x_local;

    REQUIRE_THAT(x_world.x(), WithinAbs(x_local.x() + t.x(), eps));
    REQUIRE_THAT(x_world.y(), WithinAbs(x_local.y() + t.y(), eps));
    REQUIRE_THAT(x_world.z(), WithinAbs(x_local.z() + t.z(), eps));

    // Pure rotation (matrix and quaternion) with zero translation
    const Real angle = 0.3;
    Mat3 R = Eigen::AngleAxisd(angle, Vec3::UnitY()).toRotationMatrix();
    Quat q(R); // Equivalent rotation

    Transform3 T_rot_mat = Transform3::FromRotation(R);
    Transform3 T_rot_quat = Transform3::FromRotation(q);

    const Vec3 v_local(1.0, 0.0, 0.0);

    const Vec3 v_world_ref = R * v_local;
    const Vec3 v_world_m   = T_rot_mat.rotate(v_local);
    const Vec3 v_world_q   = T_rot_quat.rotate(v_local);

    REQUIRE_THAT(v_world_m.x(), WithinAbs(v_world_ref.x(), eps));
    REQUIRE_THAT(v_world_m.y(), WithinAbs(v_world_ref.y(), eps));
    REQUIRE_THAT(v_world_m.z(), WithinAbs(v_world_ref.z(), eps));

    REQUIRE_THAT(v_world_q.x(), WithinAbs(v_world_ref.x(), eps));
    REQUIRE_THAT(v_world_q.y(), WithinAbs(v_world_ref.y(), eps));
    REQUIRE_THAT(v_world_q.z(), WithinAbs(v_world_ref.z(), eps));

    // Accessors should reflect and allow changes to R and p.
    Transform3 T = Transform3::Identity();
    T.translation() = t;
    T.rotation()    = R;

    const Vec3 x0   = Vec3::Zero();
    const Vec3 xW   = T * x0;
    const Vec3 xW_r = T.rotate(v_local);

    // Translation-only on origin should give t.
    REQUIRE_THAT(xW.x(), WithinAbs(t.x(), eps));
    REQUIRE_THAT(xW.y(), WithinAbs(t.y(), eps));
    REQUIRE_THAT(xW.z(), WithinAbs(t.z(), eps));

    // Rotation helper should still match R * v_local.
    REQUIRE_THAT(xW_r.x(), WithinAbs(v_world_ref.x(), eps));
    REQUIRE_THAT(xW_r.y(), WithinAbs(v_world_ref.y(), eps));
    REQUIRE_THAT(xW_r.z(), WithinAbs(v_world_ref.z(), eps));
}
