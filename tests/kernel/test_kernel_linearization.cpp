// Linearisation (plan task 3.7).
//
//   - a damped mass on a spring: lambda = -c/2m +- i sqrt(k/m - (c/2m)^2);
//   - a four-bar on a spring at its equilibrium, a mechanism with one degree
//     of freedom inside three coordinates: omega^2 = V'' / M along its motion
//     (from the closed-form configurations), and the shape is that motion;
//   - an operating point off the constraints, or at rest out of equilibrium,
//     is reported;
//   - a rigid body on three bushings (an engine on its mounts; plan task 5.1,
//     D42): the frequencies and mode shapes of eig(K, M), K and M in closed
//     form, and each mode's kinetic energy by coordinate (D43);
//   - a line of masses between springs: frequencies, shapes and energy
//     shares in closed form.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "mbd/core/core.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::FourBar;

namespace {

bool has_warning(const LinearModel& lm, const char* code)
{
    for (const auto& m : lm.warnings) {
        if (m.find(code) != std::string::npos) return true;
    }
    return false;
}

/// The four-bar with a spring holding it up, as in the statics test.
struct SprungFourBar {
    FourBar fb;
    Vec3 anchor{0.35, 0.9, 0.0}, on_coupler{0.4, 0.0, 0.0};
    Real k{300.0}, rest{0.2};

    SprungFourBar()
    {
        fb.sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
            0, fb.coupler, anchor, on_coupler, k, 0.0, rest));
    }

    /// Potential energy, gravity's and the spring's, at q.
    Real energy(const VecX& q) const
    {
        const Model& model = fb.sys.model;
        Data data(model);
        forward_kinematics(model, data, q);
        const Transform3& X = data.oMi[static_cast<std::size_t>(fb.coupler)];
        const Real L = (X.p + X.q * on_coupler - anchor).norm();
        return potential_energy(model, data) + 0.5 * k * (L - rest) * (L - rest);
    }
};

} // namespace

TEST_CASE("Linearisation: a damped mass on a spring", "[kernel][linearization]")
{
    // m = 2 kg, k = 800 N/m, c = 8 N s/m: w0 = 20 rad/s, zeta = c / (2 sqrt(k m))
    // = 0.1, lambda = -2 +- 19.8997 i. The forces are linear, so the central
    // differences are exact but for rounding (about 1e-15 / 2e-6 against
    // k / m = 400: relative 1e-12).
    const Real m = 2.0, k = 800.0, c = 8.0;
    System sys;
    const int body = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(),
                                        Transform3(), RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.1)),
                                        "block");
    JointCoordinateForceParams p;
    p.spring = Curve::linear(k);
    p.damper = Curve::linear(c);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, p));
    Simulator sim(sys);
    sim.q(0) = -m * g_accel / k;   // the equilibrium (gravity along -Z)

    const LinearModel lm = linearize(sim);
    INFO(lm.summary(sys.model));
    REQUIRE(lm.degrees_of_freedom == 1);
    REQUIRE(lm.modes.size() == 1);
    const Mode& mode = lm.modes[0];
    const std::complex<Real> exact(-c / (2.0 * m), std::sqrt(k / m - (c / (2.0 * m)) * (c / (2.0 * m))));
    CHECK(mode.oscillatory());
    CHECK(std::abs(mode.eigenvalue - exact) < 1e-8 * std::abs(exact));
    CHECK(std::abs(mode.natural_frequency - 20.0) < 1e-8 * 20.0);
    CHECK(std::abs(mode.damping_ratio - 0.1) < 1e-8);
    CHECK(std::abs(mode.shape(0) - 1.0) < 1e-12);
    CHECK(lm.residual < 1e-9);
    CHECK(lm.warnings.empty());
}

TEST_CASE("Linearisation: a four-bar on a spring oscillates at sqrt(V'' / M) along its motion",
          "[kernel][linearization]")
{
    // One degree of freedom in three coordinates. Along the mechanism's
    // configurations, parametrized by the crank angle th (closed form), the
    // potential V(th) has curvature V'' at the equilibrium, and the motion
    // n = dq/dth (from the loop's velocity equations) has generalized mass
    // n^T M n: omega^2 = V'' / (n^T M n). V'' by central differences of step
    // 1e-4 carries a truncation error of order h^2 V'''' / 12 and a rounding
    // error of about 1e-15 V / h^2, both near 1e-8 relative; 1e-5 on omega
    // leaves room. No damping: zeta is zero but for the differences.
    SprungFourBar sfb;
    const Model& model = sfb.fb.sys.model;
    Simulator sim(sfb.fb.sys);
    sim.q = FourBar::configuration(1.63);
    REQUIRE(static_equilibrium(sim).converged);

    const LinearModel lm = linearize(sim);
    INFO(lm.summary(model));
    REQUIRE(lm.degrees_of_freedom == 1);
    REQUIRE(lm.modes.size() == 1);
    const Mode& mode = lm.modes[0];

    const Real th = sim.q(0), hh = 1e-4;
    auto V = [&](Real crank) { return sfb.energy(FourBar::configuration(crank)); };
    const Real curvature = (V(th + hh) - 2.0 * V(th) + V(th - hh)) / (hh * hh);
    Data data(model);
    const MatX M = crba(model, data, sim.q);
    const VecX n = FourBar::motion(sim.q);
    const Real omega = std::sqrt(curvature / n.dot(M * n));
    INFO("omega from the energy " << omega << " rad/s, from the linear model " << mode.natural_frequency);
    CHECK(mode.oscillatory());
    CHECK(std::abs(mode.natural_frequency - omega) < 1e-5 * omega);
    CHECK(std::abs(mode.damping_ratio) < 1e-6);

    // The shape is the mechanism's motion, scaled to its largest entry.
    Index big = 0;
    n.cwiseAbs().maxCoeff(&big);
    const VecX n_scaled = n / n(big);
    CHECK((mode.shape - n_scaled.cast<std::complex<Real>>()).norm() < 1e-6);
}

TEST_CASE("Linearisation: an operating point off the constraints or out of equilibrium is reported",
          "[kernel][linearization]")
{
    SprungFourBar sfb;
    Simulator sim(sfb.fb.sys);
    // Off the constraints by about a centimetre, and not in equilibrium.
    sim.q = FourBar::configuration(1.0) + Eigen::Vector3d(0.0, 0.02, 0.0);
    const LinearModel lm = linearize(sim);
    INFO(lm.summary(sfb.fb.sys.model));
    CHECK(has_warning(lm, "MBD-K080"));
    CHECK(has_warning(lm, "MBD-K081"));
    CHECK(FourBar::gap(lm.q).norm() < 1e-10);   // linearised on the constraints
    CHECK((sim.q - lm.q).norm() == 0.0);        // and the simulator left there
}

// --- Plan task 5.1: modes in closed form ------------------------------------------

namespace {

/// How far a mode's shape is from the direction of phi: the part of
/// shape / |shape| not along phi, which is zero whatever scalar either was
/// multiplied by. (Normalizing both by their largest entry is ambiguous when
/// two entries are equally large, as in the symmetric modes of a line.)
Real shape_error(const Eigen::VectorXcd& shape, const VecX& phi)
{
    const Eigen::VectorXcd s = shape / shape.norm();
    const Eigen::VectorXcd u = phi.cast<std::complex<Real>>() / phi.norm();
    return (s - u * u.dot(s)).norm();
}

/// conj(phi_k) (M phi)_k / phi^H M phi for a real shape: the energy shares.
VecX energy_shares(const MatX& M, const VecX& phi)
{
    const VecX Mphi = M * phi;
    return phi.cwiseProduct(Mphi) / phi.dot(Mphi);
}

} // namespace

TEST_CASE("Linearisation: a rigid body on three bushings has the modes of eig(K, M)", "[kernel][linearization]")
{
    // A 68.5 kg engine (Seddighi's) with tilted principal axes and an offset
    // centre of mass, on three bushings to the ground, each with its own
    // stiffnesses along and about its own tilted axes, no gravity, at rest at
    // a generic pose X0. In the free joint's velocity coordinates, the body
    // frame's angular velocity and its origin's velocity d = (theta, u) in
    // body axes, a small motion moves mount i, at r_i with axes C_i (in the
    // body frame), by
    //
    //   T_i d = [C_i^T theta; C_i^T (u - [r_i]x theta)]   (rotation; displacement),
    //
    // the bushing's own coordinates to first order, so
    //
    //   K = sum_i T_i^T diag(k_rot,i, k_i) T_i,
    //   M = [[I_c - m [c]x [c]x, m [c]x], [-m [c]x, m 1]]
    //
    // (the kinetic energy m |u - [c]x theta|^2 / 2 + theta^T I_c theta / 2 at
    // rates). Without gravity the bushings carry no load at rest, so there is
    // no stiffness from a preload turning with the body and K is exact; with
    // it, linearize() would add that term and this K would not. The modes:
    // w_j^2 and phi_j from K phi = w^2 M phi, and the energy shares
    // phi_k (M phi)_k / phi^T M phi.
    //
    // Tolerances: linearize() differences the accelerations at +-1e-6 in
    // each coordinate. The model is nonlinear only through the rotations, on
    // a length scale of the mounts' offsets (0.3 m), so the differences'
    // truncation is about (1e-6 / 0.3)^2 = 1e-11 relative; their rounding is
    // that of a deflection of 1e-6 m taken from positions of about 1 m, 2e-16
    // / 1e-6 = 2e-10 relative. Frequencies are allowed 1e-8 relative; shapes
    // and energy shares, whose errors are divided by the relative gaps between
    // the squared frequencies (above 0.1 here), 1e-7.
    const Real m = 68.5;
    RigidBodyInertia body = RigidBodyInertia::from_solid_box(m, Vec3(0.25, 0.2, 0.15));
    const Mat3 tilt = Quat(Eigen::AngleAxisd(0.4, Vec3(1.0, -2.0, 0.5).normalized())).toRotationMatrix();
    body.I_com_B = tilt * body.I_com_B * tilt.transpose();
    body.com_B = Vec3(0.03, -0.02, 0.05);

    System sys;
    sys.model.gravity = Vec3::Zero();
    const int engine = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), body,
                                          "engine");
    const Transform3 X0(Quat(Eigen::AngleAxisd(0.3, Vec3::UnitZ()) * Eigen::AngleAxisd(0.2, Vec3::UnitX())),
                        Vec3(1.0, 0.5, 0.4));

    struct Mount {
        Vec3 r;
        Quat C;
        Vec3 k, k_rot;
    };
    const std::vector<Mount> mounts{
        {Vec3(0.25, 0.2, -0.1), Quat(Eigen::AngleAxisd(0.5, Vec3::UnitY())), Vec3(2.0e5, 1.5e5, 4.0e5),
         Vec3(500.0, 800.0, 300.0)},
        {Vec3(0.25, -0.2, -0.1), Quat(Eigen::AngleAxisd(-0.5, Vec3::UnitY())), Vec3(1.8e5, 1.2e5, 3.5e5),
         Vec3(400.0, 600.0, 350.0)},
        {Vec3(-0.3, 0.0, -0.05), Quat(Eigen::AngleAxisd(0.3, Vec3(1.0, 1.0, 0.0).normalized())),
         Vec3(1.0e5, 2.5e5, 3.0e5), Vec3(700.0, 300.0, 450.0)},
    };
    MatX K = MatX::Zero(6, 6);
    for (const Mount& mount : mounts) {
        BushingParams p;
        for (std::size_t a = 0; a < 3; ++a) {
            p.stiffness[a] = Curve::linear(mount.k(static_cast<Index>(a)));
            p.rotational_stiffness[a] = Curve::linear(mount.k_rot(static_cast<Index>(a)));
        }
        const Transform3 on_engine(mount.C, mount.r);
        sys.force_elements.push_back(std::make_shared<Bushing>(0, X0 * on_engine, engine, on_engine, p));

        const Mat3 Ct = mount.C.toRotationMatrix().transpose();
        Mat6 T = Mat6::Zero();
        T.topLeftCorner<3, 3>() = Ct;
        T.bottomLeftCorner<3, 3>() = -Ct * skew(mount.r);
        T.bottomRightCorner<3, 3>() = Ct;
        Vec6 k;
        k << mount.k_rot, mount.k;
        K += T.transpose() * k.asDiagonal() * T;
    }
    const Mat3 cx = skew(body.com_B);
    MatX M = MatX::Zero(6, 6);
    M.topLeftCorner(3, 3) = body.I_com_B - m * cx * cx;
    M.topRightCorner(3, 3) = m * cx;
    M.bottomLeftCorner(3, 3) = -m * cx;
    M.bottomRightCorner(3, 3) = m * Mat3::Identity();

    Simulator sim(sys);
    sim.q << X0.p, X0.q.coeffs();
    const LinearModel lm = linearize(sim);
    INFO(lm.summary(sys.model));
    REQUIRE(lm.modes.size() == 6);
    CHECK(lm.warnings.empty());

    const Eigen::GeneralizedSelfAdjointEigenSolver<MatX> closed(K, M);
    Real worst_w = 0.0, worst_shape = 0.0, worst_energy = 0.0, smallest_gap = 1.0;
    for (int j = 0; j < 6; ++j) {
        const Real w = std::sqrt(closed.eigenvalues()(j));
        const Mode& mode = lm.modes[static_cast<std::size_t>(j)];
        const VecX phi = closed.eigenvectors().col(j);
        const VecX e = energy_shares(M, phi);
        INFO("mode " << j + 1 << ": " << w / (2.0 * pi) << " Hz in closed form, " << mode.natural_frequency / (2.0 * pi)
                     << " Hz; energy shares " << e.transpose() << " against " << mode.energy_fraction.transpose());
        CHECK(mode.oscillatory());
        CHECK(std::abs(mode.damping_ratio) < 1e-8);
        CHECK(std::abs(mode.natural_frequency - w) < 1e-8 * w);
        CHECK(shape_error(mode.shape, phi) < 1e-7);
        CHECK((mode.energy_fraction - e).norm() < 1e-7);
        CHECK(std::abs(mode.energy_fraction.sum() - 1.0) < 1e-12);
        worst_w = std::max(worst_w, std::abs(mode.natural_frequency - w) / w);
        worst_shape = std::max(worst_shape, shape_error(mode.shape, phi));
        worst_energy = std::max(worst_energy, (mode.energy_fraction - e).norm());
        if (j > 0) {
            smallest_gap = std::min(smallest_gap, closed.eigenvalues()(j) / closed.eigenvalues()(j - 1) - 1.0);
        }
    }
    INFO("largest errors: frequency " << worst_w << " relative, shape " << worst_shape << ", energy shares "
                                      << worst_energy << "; smallest relative gap of w^2 " << smallest_gap);
    CHECK(smallest_gap > 0.1);
    CHECK(worst_w < 1e-8);   // repeated, to report the margins in one line
}

TEST_CASE("Linearisation: a line of masses between springs, with its energy shares", "[kernel][linearization]")
{
    // N = 4 blocks of m = 2 kg, each on its own slider along Z, in a line
    // between two walls, every neighbour (and each end to its wall) joined
    // by a spring of k = 800 N/m at its rest length; no gravity. M = m 1 and
    // K is k times the tridiagonal (2, -1), so (the standard result)
    //
    //   w_j = 2 sqrt(k / m) sin(j pi / (2 (N + 1))),  phi_j(i) = sin(i j pi / (N + 1)),
    //
    // and, M being diagonal, the share of block i in mode j's kinetic energy
    // is phi_j(i)^2 / sum_l phi_j(l)^2 = (2 / (N + 1)) sin^2(i j pi / (N + 1)).
    // The model is linear, so the differences are exact but for rounding,
    // 2e-16 of the positions (1 m) over the perturbation (1e-6 m): allowed
    // 1e-8 relative for the frequencies, 1e-7 for shapes and shares.
    const int N = 4;
    const Real m = 2.0, k = 800.0, a = 0.25;
    System sys;
    sys.model.gravity = Vec3::Zero();
    const auto slider = std::make_shared<PrismaticJointModel>();
    std::vector<int> block;
    for (int i = 1; i <= N; ++i) {
        block.push_back(sys.model.add_body(0, slider, Transform3::FromTranslation(Vec3(0.0, 0.0, i * a)), Transform3(),
                                           RigidBodyInertia::from_solid_box(m, Vec3(0.05, 0.05, 0.05)),
                                           "block " + std::to_string(i)));
    }
    const Vec3 origin = Vec3::Zero();
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0, block.front(), origin, origin, k, 0.0, a));
    for (int i = 0; i + 1 < N; ++i) {
        sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
            block[static_cast<std::size_t>(i)], block[static_cast<std::size_t>(i + 1)], origin, origin, k, 0.0, a));
    }
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        block.back(), 0, origin, Vec3(0.0, 0.0, (N + 1) * a), k, 0.0, a));

    Simulator sim(sys);
    const LinearModel lm = linearize(sim);
    INFO(lm.summary(sys.model));
    REQUIRE(lm.modes.size() == static_cast<std::size_t>(N));
    Real worst_w = 0.0, worst_shape = 0.0, worst_energy = 0.0;
    for (int j = 1; j <= N; ++j) {
        const Real w = 2.0 * std::sqrt(k / m) * std::sin(j * pi / (2.0 * (N + 1)));
        VecX phi(N), e(N);
        for (int i = 1; i <= N; ++i) {
            phi(i - 1) = std::sin(i * j * pi / (N + 1));
            e(i - 1) = 2.0 / (N + 1) * phi(i - 1) * phi(i - 1);
        }
        const Mode& mode = lm.modes[static_cast<std::size_t>(j - 1)];
        INFO("mode " << j << ": w " << mode.natural_frequency << " rad/s, closed form " << w << "; energy shares "
                     << mode.energy_fraction.transpose() << ", closed form " << e.transpose());
        CHECK(std::abs(mode.natural_frequency - w) < 1e-8 * w);
        CHECK(shape_error(mode.shape, phi) < 1e-7);
        CHECK((mode.energy_fraction - e).norm() < 1e-7);
        worst_w = std::max(worst_w, std::abs(mode.natural_frequency - w) / w);
        worst_shape = std::max(worst_shape, shape_error(mode.shape, phi));
        worst_energy = std::max(worst_energy, (mode.energy_fraction - e).norm());
    }
    INFO("largest errors: frequency " << worst_w << " relative, shape " << worst_shape << ", energy shares "
                                      << worst_energy);
    CHECK(worst_w < 1e-8);   // repeated, to report the margins in one line
}
