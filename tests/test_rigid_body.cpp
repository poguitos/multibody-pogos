#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
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
    mbd::Real mass = 2.0;
    mbd::Vec3 half_extents(0.5, 1.0, 1.5);

    mbd::RigidBodyInertia I =
        mbd::RigidBodyInertia::from_solid_box(mass, half_extents);

    REQUIRE(I.mass == mass);
    REQUIRE(I.com_B.isApprox(mbd::Vec3::Zero()));

    const mbd::Real hx = half_extents.x();
    const mbd::Real hy = half_extents.y();
    const mbd::Real hz = half_extents.z();

    const mbd::Real Ixx_exp = (mass / 3.0) * (hy * hy + hz * hz);
    const mbd::Real Iyy_exp = (mass / 3.0) * (hx * hx + hz * hz);
    const mbd::Real Izz_exp = (mass / 3.0) * (hx * hx + hy * hy);

    REQUIRE_THAT(I.I_com_B(0, 0), WithinAbs(Ixx_exp, eps));
    REQUIRE_THAT(I.I_com_B(1, 1), WithinAbs(Iyy_exp, eps));
    REQUIRE_THAT(I.I_com_B(2, 2), WithinAbs(Izz_exp, eps));

    REQUIRE_THAT(I.I_com_B(0, 1), WithinAbs(0.0, eps));
    REQUIRE_THAT(I.I_com_B(0, 2), WithinAbs(0.0, eps));
    REQUIRE_THAT(I.I_com_B(1, 2), WithinAbs(0.0, eps));

    REQUIRE(I.is_physically_valid());
}

TEST_CASE("RigidBodyInertia invalid configurations are detected",
          "[rigid_body]")
{
    mbd::RigidBodyInertia I;

    I.mass = -1.0;
    REQUIRE_FALSE(I.is_physically_valid());

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
    mbd::Quat q_raw(2.0, 0.0, 0.0, 0.0);

    mbd::RigidBodyState s = mbd::RigidBodyState::at_rest(p, q_raw);

    REQUIRE(s.p_WB.isApprox(p));
    REQUIRE(s.v_WB.isApprox(mbd::Vec3::Zero()));
    REQUIRE(s.w_WB.isApprox(mbd::Vec3::Zero()));

    REQUIRE_THAT(s.q_WB.norm(), WithinAbs(1.0, eps));

    mbd::Quat q_id = mbd::Quat::Identity();
    REQUIRE_THAT(s.q_WB.angularDistance(q_id), WithinAbs(0.0, eps));
}

TEST_CASE("RigidBodyState pose_WB returns a Transform3 consistent with p_WB and q_WB",
          "[rigid_body]")
{
    using namespace mbd;

    Vec3 p(1.0, -2.0, 0.5);
    const Real angle = 0.3;
    const Vec3 axis  = Vec3::UnitZ();
    Quat q_rot(Eigen::AngleAxisd(angle, axis));
    Quat q_raw(q_rot.coeffs() * 2.0);

    RigidBodyState s = RigidBodyState::at_rest(p, q_raw);
    Transform3 X_WB = s.pose_WB();

    // Reference: compute expected world position manually
    Mat3 R_ref = s.q_WB.toRotationMatrix();

    Vec3 x_B(0.7, -1.1, 2.3);

    Vec3 x_W     = X_WB * x_B;
    Vec3 x_W_ref = R_ref * x_B + s.p_WB;

    REQUIRE_THAT(x_W.x(), WithinAbs(x_W_ref.x(), eps));
    REQUIRE_THAT(x_W.y(), WithinAbs(x_W_ref.y(), eps));
    REQUIRE_THAT(x_W.z(), WithinAbs(x_W_ref.z(), eps));
}