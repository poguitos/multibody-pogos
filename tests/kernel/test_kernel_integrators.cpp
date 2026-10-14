// The Dormand-Prince pair (plan task 4.2), on an undamped oscillator whose
// numerical solution is known in closed form. For y' = A y every Runge-Kutta
// step multiplies the state by a polynomial in hA, its stability function;
// Dormand-Prince's is
//
//     R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 + z^5/120 + z^6/600
//
// (the series ends because the stage matrix is strictly lower triangular),
// and its error estimate, the fifth-order solution less the embedded
// fourth-order one, is E(z) y with
//
//     E(z) = -97/120000 z^5 + 13/40000 z^6 - 1/24000 z^7,
//
// both from the coefficients: b^T A^k 1 and e^T A^k 1. For the oscillator
// q'' = -q (unit mass and stiffness, w = 1 rad/s), w = q - i v obeys
// w' = i w, so z = i h, and
//
//     e^(ih) - R(ih) = h^6 / 3600 - i h^7 / 5040 + ...,
//
// an amplitude error and a phase error per step.
//
//   - one fixed step gives exactly R(ih) and E(ih);
//   - fixed steps over a span T leave an error of T (h^5 / 3600) to leading
//     order: halving the step divides it by 32, fifth order;
//   - the continuous extension is of fourth order: its error at mid-step
//     after one step falls 32-fold when the step halves;
//   - with error control the global error is proportional to the tolerance,
//     with a constant that follows from R and E;
//   - bad settings, a step below min_step, too many steps, and interpolation
//     outside the last step are refused with their codes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using Complex = std::complex<Real>;

namespace {

/// A linear spring on the slider's coordinate.
class Spring : public JointForce {
public:
    explicit Spring(Real k) : k_(k) {}
    const char* name() const override { return "spring"; }
    std::vector<int> bodies() const override { return {1}; }
    void apply(const VecX& q, const VecX&, VecX& tau) const override { tau(0) -= k_ * q(0); }

private:
    Real k_;
};

/// A 1 kg block on a slider with a spring of k N/m, no gravity, released at
/// rest from q = a.
struct Oscillator {
    static constexpr Real a = 0.1;
    System sys;

    explicit Oscillator(Real k = 1.0)
    {
        sys.model.gravity = Vec3::Zero();
        sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                           RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), "block");
        sys.joint_forces.push_back(std::make_shared<Spring>(k));
    }

    std::unique_ptr<Simulator> simulator()
    {
        auto sim = std::make_unique<Simulator>(sys);
        sim->method = Integrator::DormandPrince;
        sim->q(0) = a;
        return sim;
    }
};

/// The state as w = q - i v.
Complex w_of(const Simulator& sim) { return {sim.q(0), -sim.v(0)}; }

Complex stability(Complex z)
{
    return 1.0 + z + z * z / 2.0 + std::pow(z, 3) / 6.0 + std::pow(z, 4) / 24.0 + std::pow(z, 5) / 120.0
           + std::pow(z, 6) / 600.0;
}

Complex estimate(Complex z)
{
    return -97.0 / 120000.0 * std::pow(z, 5) + 13.0 / 40000.0 * std::pow(z, 6) - 1.0 / 24000.0 * std::pow(z, 7);
}

} // namespace

TEST_CASE("Dormand-Prince: one step is the stability polynomial, and its error estimate the embedded one",
          "[kernel][integrators]")
{
    // A long step (h = 0.3 s) so that every term of R and E counts. The error
    // measure, with relative tolerance 0 and absolute tolerance 1, is the root
    // mean square of the estimate's two entries: |E(ih)| a / sqrt(2).
    const Real h = 0.3;
    Oscillator osc;
    auto sim = osc.simulator();
    sim->step_control.adaptive = false;
    sim->step_control.relative_tolerance = 0.0;
    sim->step_control.absolute_tolerance = 1.0;
    sim->step(h);

    // Both to rounding: the state to well under 1e-13 of a (the stages
    // combine terms up to 0.35 m); the estimate (2e-7 here) is a sum of terms
    // some 10^4 times larger, so to about 1e-11 of itself.
    const Complex expected = stability(Complex(0.0, h)) * Oscillator::a;
    INFO("w = " << w_of(*sim) << ", R(ih) a = " << expected);
    CHECK(std::abs(w_of(*sim) - expected) < 1e-14);
    const Real expected_error = std::abs(estimate(Complex(0.0, h))) * Oscillator::a / std::sqrt(2.0);
    INFO("error measure " << sim->integrator_stats().error << ", expected " << expected_error);
    CHECK(std::abs(sim->integrator_stats().error - expected_error) < 1e-9 * expected_error);
    // Seven evaluations: the seventh, at the new state, is not reused across
    // calls to step().
    CHECK(sim->integrator_stats().steps == 1);
    CHECK(sim->integrator_stats().evaluations == 7);
}

TEST_CASE("Dormand-Prince: fixed steps converge at fifth order, with the error the stability function predicts",
          "[kernel][integrators]")
{
    // After N steps the state is exactly R(ih)^N a, to rounding. Against the
    // exact a e^(iT), each step loses h^6 / 3600 of amplitude and h^7 / 5040
    // of phase, so over T = 10 s the error is a T sqrt((h^5/3600)^2 +
    // (h^6/5040)^2) to leading order: 2.8e-9 m at h = 0.1 s. Halving the step
    // divides it by 32 (32.06 with the phase term).
    const Real T = 10.0;
    Oscillator osc;
    auto error_at = [&](Real h) {
        auto sim = osc.simulator();
        sim->step_control.adaptive = false;
        const int steps = sim->run(T, h);
        const Complex predicted = std::pow(stability(Complex(0.0, h)), steps) * Oscillator::a;
        INFO("h = " << h << ": w = " << w_of(*sim) << ", R(ih)^N a = " << predicted);
        CHECK(std::abs(w_of(*sim) - predicted) < 1e-13);   // rounding over 200 steps
        CHECK(sim->integrator_stats().steps == steps);
        CHECK(sim->integrator_stats().evaluations == 7 * steps);
        return std::abs(w_of(*sim) - Oscillator::a * std::exp(Complex(0.0, T)));
    };
    const Real e1 = error_at(0.1), e2 = error_at(0.05);
    auto leading = [&](Real h) {
        return Oscillator::a * T * std::hypot(std::pow(h, 5) / 3600.0, std::pow(h, 6) / 5040.0);
    };
    INFO("errors " << e1 << " and " << e2 << "; leading terms " << leading(0.1) << " and " << leading(0.05));
    CHECK(std::abs(e1 / leading(0.1) - 1.0) < 0.01);
    CHECK(std::abs(e2 / leading(0.05) - 1.0) < 0.01);
    CHECK(e1 / e2 > 31.0);
    CHECK(e1 / e2 < 33.0);
}

TEST_CASE("Dormand-Prince: the continuous extension is of fourth order", "[kernel][integrators]")
{
    // One step from the exact state: the extension's error at mid-step is of
    // order h^5 (about 2.84e-4 a h^5 here, from its coefficients), so halving
    // the step divides it by 32 (32.2 from h = 0.1 to 0.05 s, the next term
    // included). At the step's ends it is the step's own states.
    Oscillator osc;
    VecX q, v, x;
    auto midstep_error = [&](Real h) {
        auto sim = osc.simulator();
        sim->step_control.adaptive = false;
        sim->step(h);
        sim->interpolate(h, q, v, x);   // y0 + (y1 - y0): y1 to rounding
        CHECK(std::abs(q(0) - sim->q(0)) < 1e-16);
        CHECK(std::abs(v(0) - sim->v(0)) < 1e-16);
        sim->interpolate(0.0, q, v, x);
        CHECK(q(0) == Oscillator::a);
        CHECK(v(0) == 0.0);
        sim->interpolate(0.5 * h, q, v, x);
        return std::abs(Complex(q(0), -v(0)) - Oscillator::a * std::exp(Complex(0.0, 0.5 * h)));
    };
    const Real e1 = midstep_error(0.1), e2 = midstep_error(0.05);
    INFO("mid-step errors " << e1 << " and " << e2);
    CHECK(e1 / e2 > 30.5);
    CHECK(e1 / e2 < 34.0);
    CHECK(e2 < 3e-4 * Oscillator::a * std::pow(0.05, 5));
}

TEST_CASE("Dormand-Prince: with error control the global error is proportional to the tolerance",
          "[kernel][integrators]")
{
    // With relative tolerance 0 and absolute tolerance tol, a step's error
    // measure is (97/120000) h^5 a / (sqrt(2) tol) to leading order. The
    // control brings it to 0.9^5 in one step (the measure goes as h^5), so
    // h^5 = 0.9^5 sqrt(2) tol / ((97/120000) a). Each step loses a h^6 / 3600,
    // so over T the error is T a h^5 / 3600 = 0.287 T tol, whatever a. Within
    // 5 %: the phase error adds up to 1.6 % at the loosest tolerance (h =
    // 0.25 s), and the first accepted step (after rejections from h = T) and
    // the last (cut short at T) take off at most one step's share, 2.5 % of
    // 40. The steps go as tol^(-1/5): 6.3 times as many for a tolerance 10^4
    // times tighter.
    const Real T = 10.0;
    const Real constant = std::pow(0.9, 5) * std::sqrt(2.0) / (3600.0 * 97.0 / 120000.0);
    Oscillator osc;
    std::vector<long long> steps;
    for (const Real tol : {1e-7, 1e-9, 1e-11}) {
        auto sim = osc.simulator();
        sim->step_control.relative_tolerance = 0.0;
        sim->step_control.absolute_tolerance = tol;
        sim->step(T);
        const Real error = std::abs(w_of(*sim) - Oscillator::a * std::exp(Complex(0.0, T)));
        const auto& stats = sim->integrator_stats();
        INFO("tolerance " << tol << ": error " << error << ", predicted " << constant * T * tol << ", "
                          << stats.steps << " steps, " << stats.rejected << " rejected");
        CHECK(error > 0.95 * constant * T * tol);
        CHECK(error < 1.05 * constant * T * tol);
        CHECK(sim->time == T);
        steps.push_back(stats.steps);
    }
    const Real ratio = static_cast<Real>(steps[2]) / static_cast<Real>(steps[0]);
    INFO("steps " << steps[0] << ", " << steps[1] << ", " << steps[2]);
    CHECK(ratio > 5.5);
    CHECK(ratio < 7.0);
}

TEST_CASE("Dormand-Prince: bad settings, too short a step, too many steps and interpolation outside are refused",
          "[kernel][integrators]")
{
    {
        Oscillator osc;
        auto sim = osc.simulator();
        VecX q, v, x;
        CHECK_THROWS_WITH(sim->interpolate(0.0, q, v, x), ContainsSubstring("MBD-K057"));
        sim->step(0.01);
        CHECK_THROWS_WITH(sim->interpolate(0.02, q, v, x), ContainsSubstring("MBD-K057"));
        sim->step_control.absolute_tolerance = 0.0;
        CHECK_THROWS_WITH(sim->step(0.01), ContainsSubstring("MBD-K054"));
    }
    {
        // Ten periods need far more than five steps at the default tolerances.
        Oscillator osc;
        auto sim = osc.simulator();
        sim->step_control.max_steps = 5;
        CHECK_THROWS_WITH(sim->step(10.0 * 2.0 * pi), ContainsSubstring("MBD-K056"));
    }
    {
        // A spring of 1e6 N/m on 1 kg (w = 1000 rad/s): at the default
        // tolerances the velocity's error alone asks for w h of about 0.025,
        // h = 2.5e-5 s, far below a min_step of 1 ms.
        Oscillator osc(1e6);
        auto sim = osc.simulator();
        sim->step_control.min_step = 1e-3;
        CHECK_THROWS_WITH(sim->step(0.01), ContainsSubstring("MBD-K055"));
    }
}
