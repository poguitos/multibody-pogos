#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mbd/dynamics.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Semi-implicit Euler preserves constant velocity without forces",
          "[rigid_body_dynamics]")
{
    mbd::RigidBodyInertia inertia =
        mbd::RigidBodyInertia::from_solid_box(1.0, mbd::Vec3(0.5, 0.5, 0.5));

    mbd::RigidBodyState state;
    state.p_WB = mbd::Vec3::Zero();
    state.q_WB = mbd::Quat::Identity();
    state.v_WB = mbd::Vec3(1.0, -2.0, 0.5);
    state.w_WB = mbd::Vec3::Zero();

    mbd::Vec3 gravity_W = mbd::Vec3::Zero();
    mbd::RigidBodyForces forces;

    const mbd::Real dt = 0.01;
    const int steps = 100;

    const mbd::Vec3 v0 = state.v_WB;
    const mbd::Vec3 p0 = state.p_WB;
    const mbd::Quat q0 = state.q_WB;

    for (int k = 0; k < steps; ++k) {
        mbd::integrate_rigid_body_semi_implicit(inertia, gravity_W, forces, dt, state);
    }

    const mbd::Real T = steps * dt;
    const mbd::Real tol = 1e-12;

    REQUIRE(state.v_WB.isApprox(v0, tol));
    REQUIRE(state.w_WB.isApprox(mbd::Vec3::Zero(), tol));

    mbd::Vec3 p_expected = p0 + v0 * T;
    REQUIRE(state.p_WB.isApprox(p_expected, 1e-10));

    REQUIRE_THAT(state.q_WB.angularDistance(q0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Semi-implicit Euler reproduces discrete free fall with gravity only",
          "[rigid_body_dynamics]")
{
    mbd::RigidBodyInertia inertia =
        mbd::RigidBodyInertia::from_solid_box(1.0, mbd::Vec3(0.5, 0.5, 0.5));

    mbd::RigidBodyState state = mbd::RigidBodyState::at_rest(mbd::Vec3::Zero());

    mbd::Vec3 gravity_W(0.0, 0.0, -mbd::g_accel);
    mbd::RigidBodyForces forces;

    const mbd::Real dt = 0.01;

    mbd::integrate_rigid_body_semi_implicit(inertia, gravity_W, forces, dt, state);

    const mbd::Real tol = 1e-10;

    REQUIRE_THAT(state.v_WB.z(), WithinAbs(-mbd::g_accel * dt, tol));
    REQUIRE_THAT(state.p_WB.z(), WithinAbs(-mbd::g_accel * dt * dt, tol));

    REQUIRE_THAT(state.v_WB.x(), WithinAbs(0.0, tol));
    REQUIRE_THAT(state.v_WB.y(), WithinAbs(0.0, tol));
    REQUIRE_THAT(state.p_WB.x(), WithinAbs(0.0, tol));
    REQUIRE_THAT(state.p_WB.y(), WithinAbs(0.0, tol));

    REQUIRE(state.w_WB.isApprox(mbd::Vec3::Zero(), tol));
}

TEST_CASE("Gyroscopic torque is computed for non-principal axis rotation",
          "[rigid_body_dynamics]")
{
    mbd::Real mass = 1.0;
    mbd::Vec3 half_extents(0.1, 0.2, 0.3);
    mbd::RigidBodyInertia inertia = mbd::RigidBodyInertia::from_solid_box(mass, half_extents);

    mbd::RigidBodyState state;
    state.q_WB = mbd::Quat::Identity();
    state.w_WB = mbd::Vec3(1.0, 1.0, 1.0);
    state.p_WB = mbd::Vec3::Zero();
    state.v_WB = mbd::Vec3::Zero();

    mbd::RigidBodyForces forces;
    mbd::Vec3 gravity_W = mbd::Vec3::Zero();

    mbd::Vec3 a_W, alpha_W;
    mbd::compute_rigid_body_acceleration(inertia, state, forces, gravity_W, a_W, alpha_W);

    REQUIRE_THAT(a_W.norm(), WithinAbs(0.0, 1e-12));
    REQUIRE(alpha_W.norm() > 0.01);

    double Ixx = inertia.I_com_B(0, 0);
    double Iyy = inertia.I_com_B(1, 1);
    double Izz = inertia.I_com_B(2, 2);

    double expected_x = -(Izz - Iyy) / Ixx;
    double expected_y = -(Ixx - Izz) / Iyy;
    double expected_z = -(Iyy - Ixx) / Izz;

    REQUIRE_THAT(alpha_W.x(), WithinAbs(expected_x, 1e-12));
    REQUIRE_THAT(alpha_W.y(), WithinAbs(expected_y, 1e-12));
    REQUIRE_THAT(alpha_W.z(), WithinAbs(expected_z, 1e-12));
}