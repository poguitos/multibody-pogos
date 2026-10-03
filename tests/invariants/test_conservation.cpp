#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "invariants/invariant_helpers.hpp"

// Conservation laws over a simulated run, and one closed mechanical problem
// with a known answer. These catch errors that only show once the equations
// are integrated: a velocity-dependent force that is slightly wrong still
// passes a static check, but it does not conserve energy.

using namespace mbd_test;
using Catch::Matchers::WithinAbs;

namespace
{
    const JointKind kRootKinds[] = {
        JointKind::Revolute, JointKind::Prismatic, JointKind::Spherical,
        JointKind::Universal, JointKind::Free};

    const JointKind kChildKinds[] = {
        JointKind::Revolute, JointKind::Prismatic, JointKind::Spherical,
        JointKind::Universal, JointKind::Free, JointKind::Fixed};

    std::uint64_t seed_for(JointKind root, JointKind child)
    {
        return 77000u + 10u * static_cast<unsigned>(root) + static_cast<unsigned>(child);
    }
}

// ============================================================================
// Bead on a rotating rod
// ============================================================================
//
// A rod turns freely about a vertical axis through its end; a bead slides
// along it without friction. No gravity. With theta the rod angle and d the
// bead's distance from the axis:
//
//     (I0 + m d^2) theta'' = -2 m d d' theta'      (angular momentum)
//      d''                 =  d theta'^2           (centrifugal)
//
// where I0 is the inertia of rod and bead about the axis with the bead at
// d = 0. Starting from d = 0.5 m, d' = 0, theta' = 2 rad/s the bead is flung
// outward to d = 1.22427 m after one second and the rod slows to 0.50209 rad/s.

TEST_CASE("Bead on a rotating rod follows its equations of motion",
          "[invariants][conservation][prismatic]")
{
    using namespace mbd;

    const Real m_rod = 2.0, m_bead = 1.0;
    const Vec3 rod_half(0.5, 0.02, 0.02), bead_half(0.05, 0.05, 0.05);

    MultibodySystem sys;
    const BodyIndex rod = sys.add_body(
        RigidBodyInertia::from_solid_box(m_rod, rod_half), RigidBodyState{}, "rod", kGroundIndex);
    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(), Transform3::Identity(), kGroundIndex, rod));

    // Prismatic joints slide along their local Z; turn Z onto the rod's X.
    const Mat3 R = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitY()).toRotationMatrix();
    const BodyIndex bead = sys.add_body(
        RigidBodyInertia::from_solid_box(m_bead, bead_half), RigidBodyState{}, "bead", rod);
    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        Transform3(R, Vec3::Zero()), Transform3::FromRotation(R), rod, bead));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sys.q << 0.0, 0.5;
    sys.q_dot << 2.0, 0.0;
    sys.compute_kinematics();

    const Real E0 = kinetic_energy(sys);
    sim.run(1.0, 1e-4);
    const Real E1 = kinetic_energy(sys);

    // Reference: the two equations above, integrated with RK4 at 1e-5 s.
    // I0: box inertias about Z, m/3 * (hx^2 + hy^2), for rod and bead.
    const Real I0 = m_rod / 3.0 * (rod_half.x() * rod_half.x() + rod_half.y() * rod_half.y())
                  + m_bead / 3.0 * (bead_half.x() * bead_half.x() + bead_half.y() * bead_half.y());
    Real y[4] = {0.0, 0.5, 2.0, 0.0};   // theta, d, theta', d'
    auto f = [&](const Real* a, Real* out) {
        out[0] = a[2];
        out[1] = a[3];
        out[2] = -2.0 * m_bead * a[1] * a[3] * a[2] / (I0 + m_bead * a[1] * a[1]);
        out[3] = a[1] * a[2] * a[2];
    };
    const Real h = 1e-5;
    for (int i = 0; i < 100000; ++i) {
        Real k1[4], k2[4], k3[4], k4[4], t[4];
        f(y, k1);
        for (int j = 0; j < 4; ++j) t[j] = y[j] + 0.5 * h * k1[j];
        f(t, k2);
        for (int j = 0; j < 4; ++j) t[j] = y[j] + 0.5 * h * k2[j];
        f(t, k3);
        for (int j = 0; j < 4; ++j) t[j] = y[j] + h * k3[j];
        f(t, k4);
        for (int j = 0; j < 4; ++j) y[j] += h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    }

    INFO("engine d = " << sys.q(1) << " m, reference d = " << y[1] << " m");
    CHECK_THAT(y[1], WithinAbs(1.22427, 1e-5));        // the reference itself
    CHECK_THAT(sys.q(1), WithinAbs(y[1], 1e-8));       // bead position
    CHECK_THAT(sys.q_dot(0), WithinAbs(y[2], 1e-8));   // rod angular velocity
    CHECK_THAT(sys.q(0), WithinAbs(y[0], 1e-8));       // rod angle
    CHECK_THAT(E1, WithinAbs(E0, 1e-9));               // nothing dissipates
}

// ============================================================================
// Energy
// ============================================================================

TEST_CASE("Invariant: kinetic plus potential energy is conserved under gravity",
          "[invariants][conservation]")
{
    const Vec3 gravity(0.0, -mbd::g_accel, 0.0);

    for (JointKind root : kRootKinds) {
        for (JointKind child : kChildKinds) {
            DYNAMIC_SECTION(joint_name(root) << " -> " << joint_name(child)) {
                Rng rng(seed_for(root, child));
                MultibodySystem sys;
                build_chain(sys, rng, root, child);

                mbd::Simulator sim(sys);
                sim.set_gravity(gravity);
                sim.method = mbd::IntegrationMethod::RK4;
                sim.initialize();
                randomize_state(sys, rng);

                const Real T0 = kinetic_energy(sys);
                const Real V0 = potential_energy(sys, gravity);
                sim.run(0.5, 5e-4);
                sys.compute_kinematics();
                const Real T1 = kinetic_energy(sys);
                const Real V1 = potential_energy(sys, gravity);

                // RK4 at this step conserves energy to about 1e-10 relative;
                // a wrong velocity-dependent force shows up at 1e-3 or more.
                const Real scale = 1.0 + std::abs(T0) + std::abs(V0) + std::abs(T1);
                CAPTURE(T0, V0, T1, V1);
                CHECK(std::abs((T1 + V1) - (T0 + V0)) < 1e-7 * scale);
            }
        }
    }
}

// ============================================================================
// Momentum
// ============================================================================

TEST_CASE("Invariant: a free-floating chain conserves linear and angular momentum",
          "[invariants][conservation]")
{
    for (JointKind child : kChildKinds) {
        DYNAMIC_SECTION("free -> " << joint_name(child)) {
            Rng rng(seed_for(JointKind::Free, child) + 500u);
            MultibodySystem sys;
            build_chain(sys, rng, JointKind::Free, child);

            mbd::Simulator sim(sys);
            sim.set_gravity(Vec3::Zero());
            sim.method = mbd::IntegrationMethod::RK4;
            sim.initialize();
            randomize_state(sys, rng);

            const Vec3 P0 = linear_momentum(sys);
            const Vec3 H0 = angular_momentum(sys);
            sim.run(0.5, 5e-4);
            sys.compute_kinematics();
            const Vec3 P1 = linear_momentum(sys);
            const Vec3 H1 = angular_momentum(sys);

            // No external force or torque acts on the chain as a whole.
            CAPTURE((P1 - P0).norm(), (H1 - H0).norm());
            CHECK((P1 - P0).norm() < 1e-7 * (1.0 + P0.norm()));
            CHECK((H1 - H0).norm() < 1e-7 * (1.0 + H0.norm()));
        }
    }
}
