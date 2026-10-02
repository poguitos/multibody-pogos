// Tests for rotation-vector canonicalization of free/spherical joints.
//
// Free and spherical joints parameterize orientation with a rotation vector
// (exponential map). Its Jacobian degenerates at ||r|| = 2*pi, which used to
// drive the generalized mass matrix non-SPD once a body accumulated enough
// rotation. Canonicalization keeps ||r|| <= pi every step while preserving the
// physical orientation and angular velocity, so a body can spin indefinitely.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mbd/model/system.hpp"
#include "mbd/model/joint.hpp"
#include "mbd/integrators/simulator.hpp"

using namespace mbd;
using Catch::Matchers::WithinRel;

namespace {

/// Rotational kinetic energy and angular momentum (world frame) of a body.
struct RotInvariants { Real KE; Vec3 L; };

RotInvariants rot_invariants(const MultibodySystem& sys, BodyIndex b)
{
    const Mat3 R   = sys.states[b].q_WB.toRotationMatrix();
    const Mat3 I_W = R * sys.inertias[b].I_com_B * R.transpose();
    const Vec3 w   = sys.states[b].w_WB;
    return { Real(0.5) * w.dot(I_W * w), I_W * w };
}

} // namespace

TEST_CASE("Orientation: asymmetric free body spins indefinitely without singularity",
          "[orientation][canonicalization][free]")
{
    MultibodySystem sys;
    // Asymmetric inertia => torque-free precession: the angular-velocity
    // direction changes over time, exciting the off-axis exp-map subspace as
    // ||r|| accumulates. This is exactly the case that used to blow up.
    auto inertia = RigidBodyInertia::from_solid_box(2.0, Vec3(0.5, 0.3, 0.15));
    BodyIndex b = sys.add_body(inertia, RigidBodyState{}, "b", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, b));

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(3) = 6.0;   // dominant spin
    sys.q_dot(4) = 0.7;   // off-axis components -> genuine 3D precession
    sys.q_dot(5) = 0.4;
    sys.compute_kinematics();

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    const RotInvariants inv0 = rot_invariants(sys, b);

    Real max_r = 0.0;
    for (int i = 0; i < 5000; ++i) {        // 5 s, many turns past 2*pi
        sim.step(0.001);                    // must not throw
        max_r = std::max(max_r, sys.q.segment<3>(3).norm());
    }

    const RotInvariants inv1 = rot_invariants(sys, b);

    INFO("max |r| = " << max_r);
    // Canonicalization kept the rotation vector bounded (previously unbounded).
    REQUIRE(max_r <= pi + 1e-9);
    // Torque-free motion conserves rotational KE and angular-momentum magnitude.
    REQUIRE_THAT(inv1.KE,       WithinRel(inv0.KE,       0.02));
    REQUIRE_THAT(inv1.L.norm(), WithinRel(inv0.L.norm(), 0.02));
}

TEST_CASE("Orientation: spherical joint spins past 2*pi without singularity",
          "[orientation][canonicalization][spherical]")
{
    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.3, 0.2, 0.1));
    BodyIndex b = sys.add_body(inertia, RigidBodyState{}, "b", kGroundIndex);
    sys.add_joint(std::make_unique<SphericalCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, b));

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.q_dot(0) = 5.0;
    sys.q_dot(1) = 0.6;
    sys.q_dot(2) = 0.3;
    sys.compute_kinematics();

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    Real max_r = 0.0;
    for (int i = 0; i < 3000; ++i) {
        sim.step(0.001);                    // must not throw
        max_r = std::max(max_r, sys.q.head<3>().norm());
    }

    REQUIRE(max_r <= pi + 1e-9);
    // The body kept spinning (did not lock up at the singularity).
    REQUIRE(sys.states[b].w_WB.norm() > 1.0);
}
