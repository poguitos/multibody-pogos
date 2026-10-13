// Linearisation (plan task 3.7).
//
//   - a damped mass on a spring: lambda = -c/2m +- i sqrt(k/m - (c/2m)^2);
//   - a four-bar on a spring at its equilibrium, a mechanism with one degree
//     of freedom inside three coordinates: omega^2 = V'' / M along its motion
//     (from the closed-form configurations), and the shape is that motion;
//   - an operating point off the constraints, or at rest out of equilibrium,
//     is reported.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mbd/core/core.hpp"
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
