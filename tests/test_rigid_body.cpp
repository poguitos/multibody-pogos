#include <catch2/catch_all.hpp>
#include <Eigen/Geometry>

#include "mbd/math.hpp"
#include "mbd/rigid_body.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-12;
}

TEST_CASE("RigidBodyInertia from_solid_box computes correct diagonal inertia",
          "[rigid_body]")
{
    // Box with half extents (hx, hy, hz)
    mbd::Real mass = 2.0;
    mbd::Vec3 half_extents(0.5, 1.0, 1.5);

    mbd::RigidBodyInertia I =
        mbd::RigidBodyInertia::from_solid_box(mass, half_extents);

    REQUIRE(I.mass == mass);
    REQUIRE(I.com_B.isApprox(mbd::Vec3::Zero()));

    // Expected moments:
    // Ixx = (1/3) * m * (hy^2 + hz^2)
    // Iyy = (1/3) * m * (hx^2 + hz^2)
    // Izz = (1/3) * m * (hx^2 + hy^2)
    const mbd::Real hx = half_extents.x();
    const mbd::Real hy = half_extents.y();
    const mbd::Real hz = half_extents.z();

    const mbd::Real Ixx_exp = (mass / 3.0) * (hy * hy + hz * hz);
    const mbd::Real Iyy_exp = (mass / 3.0) * (hx * hx + hz * hz);
    const mbd::Real Izz_exp = (mass / 3.0) * (hx * hx + hy * hy);

    const mbd::Real tol = 1e-12;

    REQUIRE_THAT(I.I_com_B(0, 0), WithinAbs(Ixx_exp, tol));
    REQUIRE_THAT(I.I_com_B(1, 1), WithinAbs(Iyy_exp, tol));
    REQUIRE_THAT(I.I_com_B(2, 2), WithinAbs(Izz_exp, tol));

    // Off-diagonals should be (approximately) zero for this symmetric box
    REQUIRE_THAT(I.I_com_B(0, 1), WithinAbs(0.0, tol));
    REQUIRE_THAT(I.I_com_B(0, 2), WithinAbs(0.0, tol));
    REQUIRE_THAT(I.I_com_B(1, 2), WithinAbs(0.0, tol));

    REQUIRE(I.is_physically_valid());
}

TEST_CASE("RigidBodyInertia invalid configurations are detected",
          "[rigid_body]")
{
    mbd::RigidBodyInertia I;

    // Negative mass is invalid
    I.mass = -1.0;
    REQUIRE_FALSE(I.is_physically_valid());

    // Non-symmetric inertia matrix is invalid
    I.mass   = 1.0;
    I.com_B  = mbd::Vec3::Zero();
    I.I_com_B << 1.0, 2.0, 3.0,
                 0.0, 1.0, 4.0,
                 0.0, 0.0, 1.0;

    REQUIRE_FALSE(I.is_physically_valid());
}

TEST_CASE("RigidBodyState at_rest sets zero velocities and normalizes quaternion",
          "[rigid_body]")
{
    mbd::Vec3 p(1.0, 2.0, 3.0);

    // Non-unit quaternion on purpose to test normalization
    mbd::Quat q_raw(2.0, 0.0, 0.0, 0.0); // equivalent to identity after normalization

    mbd::RigidBodyState s = mbd::RigidBodyState::at_rest(p, q_raw);

    REQUIRE(s.p_WB.isApprox(p));
    REQUIRE(s.v_WB.isApprox(mbd::Vec3::Zero()));
    REQUIRE(s.w_WB.isApprox(mbd::Vec3::Zero()));

    // Orientation must be normalized
    const double norm = s.q_WB.norm();
    REQUIRE_THAT(norm, WithinAbs(1.0, eps));

    // q_raw was along the scalar component only -> same rotation as identity
    mbd::Quat q_id = mbd::Quat::Identity();
    REQUIRE_THAT(s.q_WB.angularDistance(q_id), WithinAbs(0.0, eps));
}

TEST_CASE("RigidBodyState pose_WB returns a Transform3 consistent with p_WB and q_WB",
          "[rigid_body]")
{
    using namespace mbd;

    // Position and orientation for the body frame
    Vec3 p(1.0, -2.0, 0.5);

    const Real angle = 0.3;           // rad
    const Vec3 axis  = Vec3::UnitZ(); // rotate about Z

    Quat q_rot(Eigen::AngleAxisd(angle, axis));

    // Deliberately scale the quaternion to test normalization via at_rest
    Quat q_raw(q_rot.coeffs() * 2.0);

    RigidBodyState s = RigidBodyState::at_rest(p, q_raw);

    // Pose from the new API
    Transform3 X_WB = s.pose_WB();

    // Reference pose built directly from q_WB and p_WB
    Mat3 R_ref = s.q_WB.toRotationMatrix();
    Transform3 X_ref(R_ref, s.p_WB);

    // Test point in body frame
    Vec3 x_B(0.7, -1.1, 2.3);

    Vec3 x_W     = X_WB * x_B;
    Vec3 x_W_ref = X_ref * x_B;

    REQUIRE_THAT(x_W.x(), WithinAbs(x_W_ref.x(), eps));
    REQUIRE_THAT(x_W.y(), WithinAbs(x_W_ref.y(), eps));
    REQUIRE_THAT(x_W.z(), WithinAbs(x_W_ref.z(), eps));
}
