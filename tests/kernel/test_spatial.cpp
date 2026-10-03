// Spatial algebra (include/mbd/spatial/spatial.hpp): the identities that the
// recursive algorithms rely on, each checked on random frames and vectors.

#include <catch2/catch_test_macros.hpp>

#include "mbd/spatial/spatial.hpp"

#include "support/rng.hpp"

using namespace mbd;
using mbd_test::Rng;

namespace {

constexpr Real kTol = 1e-12;

Vec6 random_vec6(Rng& rng)
{
    Vec6 m;
    m << rng.vec(2.0), rng.vec(2.0);
    return m;
}

RigidBodyInertia random_inertia(Rng& rng)
{
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(
        rng.range(0.5, 5.0), Vec3(rng.range(0.05, 0.5), rng.range(0.05, 0.5), rng.range(0.05, 0.5)));
    // A general orientation of the principal axes and an offset centre of mass.
    const Mat3 R = rng.quat().toRotationMatrix();
    I.I_com_B = R * I.I_com_B * R.transpose();
    I.com_B   = rng.vec(0.4);
    return I;
}

} // namespace

TEST_CASE("Spatial: changes of frame are inverse to each other", "[kernel][spatial]")
{
    Rng rng(1);
    for (int trial = 0; trial < 20; ++trial) {
        const Transform3 X = rng.frame(1.5);
        const Vec6 m = random_vec6(rng);
        const Vec6 f = random_vec6(rng);
        CHECK((motion_act_inv(X, motion_act(X, m)) - m).norm() < kTol);
        CHECK((motion_act(X, motion_act_inv(X, m)) - m).norm() < kTol);
        CHECK((force_act_inv(X, force_act(X, f)) - f).norm() < kTol);
        CHECK((force_act(X, force_act_inv(X, f)) - f).norm() < kTol);
        // The inverse of the frame does the inverse change.
        CHECK((motion_act(X.inverse(), m) - motion_act_inv(X, m)).norm() < kTol);
        CHECK((force_act(X.inverse(), f) - force_act_inv(X, f)).norm() < kTol);
    }
}

TEST_CASE("Spatial: matrix forms match the functions", "[kernel][spatial]")
{
    Rng rng(2);
    for (int trial = 0; trial < 20; ++trial) {
        const Transform3 X = rng.frame(1.5);
        const Vec6 m = random_vec6(rng);
        const Vec6 f = random_vec6(rng);
        CHECK((motion_matrix(X) * m - motion_act(X, m)).norm() < kTol);
        CHECK((force_matrix(X) * f - force_act(X, f)).norm() < kTol);
        CHECK((force_matrix(X) - motion_matrix(X).inverse().transpose()).norm() < 1e-11);
    }
}

TEST_CASE("Spatial: power is the same in every frame", "[kernel][spatial]")
{
    Rng rng(3);
    for (int trial = 0; trial < 20; ++trial) {
        const Transform3 X = rng.frame(1.5);
        const Vec6 m = random_vec6(rng);
        const Vec6 f = random_vec6(rng);
        CHECK(std::abs(force_act(X, f).dot(motion_act(X, m)) - f.dot(m)) < kTol);
    }
}

TEST_CASE("Spatial: changes of frame compose", "[kernel][spatial]")
{
    Rng rng(4);
    for (int trial = 0; trial < 20; ++trial) {
        const Transform3 X_AB = rng.frame(1.5);
        const Transform3 X_BC = rng.frame(1.5);
        const Vec6 m = random_vec6(rng);
        const Vec6 f = random_vec6(rng);
        CHECK((motion_act(X_AB * X_BC, m) - motion_act(X_AB, motion_act(X_BC, m))).norm() < kTol);
        CHECK((force_act(X_AB * X_BC, f) - force_act(X_AB, force_act(X_BC, f))).norm() < kTol);
    }
}

TEST_CASE("Spatial: a motion vector is the velocity field of a rigid body", "[kernel][spatial]")
{
    // The motion vector in A of a body with angular velocity w and origin
    // velocity v_O (both in B) gives, for any point P, v_P = v + w x P in A.
    Rng rng(5);
    const Transform3 X_AB = rng.frame(1.5);
    const Vec3 w_B = rng.vec(2.0), vO_B = rng.vec(2.0);
    Vec6 m_B;
    m_B << w_B, vO_B;
    const Vec6 m_A = motion_act(X_AB, m_B);

    const Vec3 P_B = rng.vec(1.0);
    const Vec3 vP_B = vO_B + w_B.cross(P_B);
    const Vec3 P_A = X_AB.apply(P_B);
    const Vec3 vP_A = m_A.tail<3>() + m_A.head<3>().cross(P_A);
    CHECK((vP_A - X_AB.q * vP_B).norm() < kTol);
}

TEST_CASE("Spatial: cross products", "[kernel][spatial]")
{
    Rng rng(6);
    for (int trial = 0; trial < 20; ++trial) {
        const Transform3 X = rng.frame(1.5);
        const Vec6 m1 = random_vec6(rng);
        const Vec6 m2 = random_vec6(rng);
        const Vec6 f  = random_vec6(rng);

        // Duality: (m1 x m2) . f + m2 . (m1 x* f) = 0.
        CHECK(std::abs(motion_cross_motion(m1, m2).dot(f) + m2.dot(motion_cross_force(m1, f))) < 1e-11);
        // Antisymmetry and m x m = 0.
        CHECK((motion_cross_motion(m1, m2) + motion_cross_motion(m2, m1)).norm() < kTol);
        CHECK(motion_cross_motion(m1, m1).norm() < kTol);
        // Both commute with changes of frame.
        CHECK((motion_act(X, motion_cross_motion(m1, m2))
               - motion_cross_motion(motion_act(X, m1), motion_act(X, m2))).norm() < 1e-11);
        CHECK((force_act(X, motion_cross_force(m1, f))
               - motion_cross_force(motion_act(X, m1), force_act(X, f))).norm() < 1e-11);
    }
}

TEST_CASE("Spatial: rigid-body inertia", "[kernel][spatial]")
{
    Rng rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        const RigidBodyInertia body = random_inertia(rng);
        const Mat6 I = spatial_inertia_matrix(body);
        const Vec6 m = random_vec6(rng);

        // Symmetric and positive definite.
        CHECK((I - I.transpose()).norm() < kTol);
        CHECK(Eigen::SelfAdjointEigenSolver<Mat6>(I).eigenvalues().minCoeff() > 0.0);

        // Matrix-free product.
        CHECK((spatial_inertia_times(body, m) - I * m).norm() < 1e-11);

        // Kinetic energy from the centre-of-mass motion: m |v_c|^2 / 2 + w.I_c w / 2.
        const Vec3 w = m.head<3>();
        const Vec3 v_com = m.tail<3>() + w.cross(body.com_B);
        const Real T = 0.5 * body.mass * v_com.squaredNorm() + 0.5 * w.dot(body.I_com_B * w);
        CHECK(std::abs(0.5 * m.dot(I * m) - T) < 1e-11);
    }
}

TEST_CASE("Spatial: inertia in another frame is the body described in that frame",
          "[kernel][spatial]")
{
    // Moving the 6x6 inertia to frame A must give the matrix built from the
    // centre of mass and rotational inertia re-expressed in A: this checks
    // inertia_act and the parallel-axis terms of spatial_inertia_matrix.
    Rng rng(8);
    for (int trial = 0; trial < 20; ++trial) {
        const RigidBodyInertia body_B = random_inertia(rng);
        const Transform3 X_AB = rng.frame(1.5);
        const Mat3 R = X_AB.rotation_matrix();
        const RigidBodyInertia body_A(body_B.mass, X_AB.apply(body_B.com_B),
                                      R * body_B.I_com_B * R.transpose());

        const Mat6 I_B = spatial_inertia_matrix(body_B);
        const Mat6 I_A = inertia_act(X_AB, I_B);
        CHECK((I_A - spatial_inertia_matrix(body_A)).norm() < 1e-10);

        // Momentum transforms as a force: I_A (X m) = X* (I_B m).
        const Vec6 m = random_vec6(rng);
        CHECK((I_A * motion_act(X_AB, m) - force_act(X_AB, I_B * m)).norm() < 1e-10);
    }
}

TEST_CASE("Spatial: exp3 and log3 are inverse", "[kernel][spatial]")
{
    Rng rng(9);
    for (Real angle : {0.0, 1e-12, 1e-7, 1e-3, 0.5, 2.0, 3.1}) {
        for (int trial = 0; trial < 5; ++trial) {
            const Vec3 w = angle * rng.direction();
            const Quat Q = exp3(w);
            CHECK(std::abs(Q.norm() - 1.0) < 1e-15);
            CHECK((log3(Q) - w).norm() < 1e-14 * (1.0 + angle));
            // The rotation by |w| about w.
            const Vec3 axis = angle > 0.0 ? Vec3(w.normalized()) : Vec3(Vec3::UnitX());
            CHECK(Q.angularDistance(Quat(Eigen::AngleAxisd(angle, axis))) < 1e-14);
            // Both signs of the quaternion give the same rotation vector.
            CHECK((log3(Quat(-Q.coeffs())) - w).norm() < 1e-14 * (1.0 + angle));
        }
    }
}

TEST_CASE("Spatial: exp6 and log6 are inverse", "[kernel][spatial]")
{
    Rng rng(10);
    for (Real angle : {0.0, 1e-9, 1e-4, 9e-3, 1.1e-2, 0.5, 2.0, 3.1}) {
        for (int trial = 0; trial < 5; ++trial) {
            Vec6 xi;
            xi << angle * rng.direction(), rng.vec(2.0);
            const Transform3 X = exp6(xi);
            CHECK((log6(X) - xi).norm() < 1e-13 * (1.0 + xi.norm()));

            const Transform3 Y = rng.frame(2.0);
            const Transform3 Z = exp6(log6(Y));
            CHECK((Z.p - Y.p).norm() < 1e-13);
            CHECK(Z.q.angularDistance(Y.q) < 1e-14);
        }
    }
}

TEST_CASE("Spatial: exp6 is the motion at constant body-fixed velocity", "[kernel][spatial]")
{
    // The motions exp6(s xi) form a one-parameter group, and for small s the
    // frame moves by s v along its own axes: together these say that exp6(xi)
    // is reached by moving at the constant body-fixed velocity xi.
    Rng rng(11);
    for (int trial = 0; trial < 10; ++trial) {
        Vec6 xi;
        xi << rng.vec(2.0), rng.vec(2.0);
        const Transform3 half = exp6(0.5 * xi);
        const Transform3 full = exp6(xi);
        const Transform3 twice = half * half;
        CHECK((twice.p - full.p).norm() < 1e-14 * (1.0 + full.p.norm()));
        CHECK(twice.q.angularDistance(full.q) < 1e-14);

        const Real s = 1e-7;
        const Transform3 small = exp6(s * xi);
        CHECK((small.p / s - xi.tail<3>()).norm() < 1e-5);
        CHECK((log3(small.q) / s - xi.head<3>()).norm() < 1e-9);
    }
}
