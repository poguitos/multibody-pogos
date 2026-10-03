#include <catch2/catch_test_macros.hpp>

#include "invariants/invariant_helpers.hpp"

// Every joint type is tested as the root of a chain and as the child of a
// moving parent, at several generic states. The chain is
//
//     ground -> (root joint) -> (child joint) -> revolute
//
// so that an error made at the second joint is also seen further down.
// Joint frames are randomly rotated and offset: nothing here relies on a joint
// axis being aligned with a world axis or on a frame sitting at a body origin.

using namespace mbd_test;

namespace
{
    const JointKind kRootKinds[] = {
        JointKind::Revolute, JointKind::Prismatic, JointKind::Spherical,
        JointKind::Universal, JointKind::Free};

    const JointKind kChildKinds[] = {
        JointKind::Revolute, JointKind::Prismatic, JointKind::Spherical,
        JointKind::Universal, JointKind::Free, JointKind::Fixed};

    constexpr int kSeeds = 3;

    // Central differences with a step of 1e-6 are accurate to about 1e-9 here;
    // a real defect shows up at 1e-2 or more.
    constexpr Real kTol = 1e-6;

    std::uint64_t seed_for(int seed, JointKind root, JointKind child)
    {
        return 1000u * static_cast<unsigned>(seed)
             + 10u * static_cast<unsigned>(root)
             + static_cast<unsigned>(child);
    }
}

TEST_CASE("Invariant: body velocities are the time derivative of body poses",
          "[invariants][kinematics]")
{
    for (JointKind root : kRootKinds) {
        for (JointKind child : kChildKinds) {
            DYNAMIC_SECTION(joint_name(root) << " -> " << joint_name(child)) {
                for (int seed = 1; seed <= kSeeds; ++seed) {
                    Rng rng(seed_for(seed, root, child));
                    MultibodySystem sys;
                    build_chain(sys, rng, root, child);
                    randomize_state(sys, rng);

                    const KinematicsErrors e = kinematics_errors(sys);
                    CAPTURE(seed, e.v_fk, e.w_fk, e.v_jac, e.w_jac);

                    // Velocity pass (states[i].v_WB, w_WB)
                    CHECK(e.v_fk < kTol * e.scale);
                    CHECK(e.w_fk < kTol * e.scale);
                    // Body Jacobian times q_dot
                    CHECK(e.v_jac < kTol * e.scale);
                    CHECK(e.w_jac < kTol * e.scale);
                }
            }
        }
    }
}

TEST_CASE("Invariant: inverse dynamics, mass matrix and Lagrange's equations agree",
          "[invariants][dynamics]")
{
    for (JointKind root : kRootKinds) {
        for (JointKind child : kChildKinds) {
            DYNAMIC_SECTION(joint_name(root) << " -> " << joint_name(child)) {
                for (int seed = 1; seed <= kSeeds; ++seed) {
                    Rng rng(seed_for(seed, root, child));
                    MultibodySystem sys;
                    build_chain(sys, rng, root, child);
                    randomize_state(sys, rng);

                    const DynamicsErrors e = dynamics_errors(sys, rng);
                    CAPTURE(seed, e.rnea_vs_mass, e.bias_vs_lagrange, e.gravity_vs_potential);

                    // The acceleration-dependent part of inverse dynamics is
                    // exactly the mass matrix times the acceleration.
                    CHECK(e.rnea_vs_mass < kTol * e.scale_mass);
                    // The velocity-dependent part is the one Lagrange's
                    // equations give for that mass matrix. The reference is a
                    // finite difference of M, hence the looser tolerance.
                    CHECK(e.bias_vs_lagrange < 1e-5 * e.scale_bias);
                    // The gravity part is the gradient of the potential.
                    CHECK(e.gravity_vs_potential < kTol * e.scale_gravity);
                }
            }
        }
    }
}
