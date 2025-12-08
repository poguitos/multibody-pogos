#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include <mbd/dynamics.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Approx;


TEST_CASE("Semi-implicit Euler preserves constant velocity without forces", "[rigid_body_dynamics]")
{
    // Inertia: any valid inertia, e.g. unit mass solid box
    mbd::RigidBodyInertia inertia =
        mbd::RigidBodyInertia::from_solid_box(
            1.0, mbd::Vec3(0.5, 0.5, 0.5));

    // Initial state: position at origin, constant velocity, no rotation
    mbd::RigidBodyState state;
    state.p_WB = mbd::Vec3::Zero();
    state.q_WB = mbd::Quat::Identity();
    state.v_WB = mbd::Vec3(1.0, -2.0, 0.5);
    state.w_WB = mbd::Vec3::Zero();

    mbd::Vec3 gravity_W = mbd::Vec3::Zero();
    mbd::RigidBodyForces forces; // zero forces and torques

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

    // Velocities should remain constant
    REQUIRE(state.v_WB.isApprox(v0, tol));
    REQUIRE(state.w_WB.isApprox(mbd::Vec3::Zero(), tol));

    // Position should follow p = p0 + v * T
    mbd::Vec3 p_expected = p0 + v0 * T;
    REQUIRE(state.p_WB.isApprox(p_expected, 1e-10));

    // Orientation should remain identity (no angular velocity)
    REQUIRE(state.q_WB.angularDistance(q0) == Approx(0.0).margin(1e-12));
}

TEST_CASE("Semi-implicit Euler reproduces discrete free fall with gravity only", "[rigid_body_dynamics]")
{
    // Inertia: unit mass, inertia does not matter for translation without torque
    mbd::RigidBodyInertia inertia =
        mbd::RigidBodyInertia::from_solid_box(
            1.0, mbd::Vec3(0.5, 0.5, 0.5));

    // Start at rest at origin
    mbd::RigidBodyState state = mbd::RigidBodyState::at_rest(mbd::Vec3::Zero());

    // Gravity pointing in -Z
    mbd::Vec3 gravity_W(0.0, 0.0, -mbd::g);

    mbd::RigidBodyForces forces; // no extra forces or torques

    const mbd::Real dt = 0.01;

    // One integration step
    mbd::integrate_rigid_body_semi_implicit(inertia, gravity_W, forces, dt, state);

    const mbd::Real tol = 1e-10;

    // Semi-implicit Euler:
    // v1 = v0 + g*dt = g*dt
    // p1 = p0 + v1*dt = g*dt^2
    REQUIRE_THAT(state.v_WB.z(), WithinAbs(-mbd::g * dt, tol));
    REQUIRE_THAT(state.p_WB.z(), WithinAbs(-mbd::g * dt * dt, tol));

    // X and Y should stay zero (no gravity in those directions)
    REQUIRE_THAT(state.v_WB.x(), WithinAbs(0.0, tol));
    REQUIRE_THAT(state.v_WB.y(), WithinAbs(0.0, tol));
    REQUIRE_THAT(state.p_WB.x(), WithinAbs(0.0, tol));
    REQUIRE_THAT(state.p_WB.y(), WithinAbs(0.0, tol));

    // No torque -> no angular acceleration
    REQUIRE(state.w_WB.isApprox(mbd::Vec3::Zero(), tol));
}
