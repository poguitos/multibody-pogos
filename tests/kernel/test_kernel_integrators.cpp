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
//
// The generalised-alpha method (plan task 4.3), on the same oscillator and on
// a stiff two-mass system, against the method's own recurrence written out
// for a linear system (q'' = -w^2 q, a0 = qdd0):
//
//     a+ = (-(1 - af) w^2 (q + h v + h^2 (1/2 - beta) a) - am a + af qdd)
//          / ((1 - am) + (1 - af) beta h^2 w^2)
//     q+ = q + h v + h^2 ((1/2 - beta) a + beta a+),  v+ = v + h ((1 - gamma) a + gamma a+)
//
//   - the simulator's steps are this recurrence, to Newton's tolerance, at
//     second order;
//   - a spring 1000 times too stiff for RK4 at 1 ms (RK4 needs w h <= 2
//     sqrt(2): |R(iy)|^2 = 1 - y^6/72 + y^8/576) runs stably, damped as
//     rho_infinity says, or energy-conserving at rho_infinity = 1;
//   - with such a spring linking two masses, the slow motion keeps the
//     method's second-order accuracy;
//   - bad settings and a step that cannot be solved are refused.
//
// SDIRK4 (plan task 4.4), whose step on a linear system multiplies the state
// by its stability function R(z) = 1 + z b^T (I - z A)^-1 1, computed here
// from its coefficients (fourth order: e^z - R(z) = 8.45e-4 z^5 + ...; L-stable:
// R(z) -> 0 as z -> infinity, like 9.3 / z):
//
//   - one step is R(ih), fixed steps converge at fourth order;
//   - a spring far too stiff for the step shrinks by |R(i w h)| in one step;
//   - with error control on the stiff two-mass system the steps follow the
//     slow motion, not the stiff one.
//
// Radau IIA (task 4.4, D34), checked the same way: R(z) from its
// coefficients (the (2, 3) Pade approximant: e^z - R(z) = -z^6 / 7200 + ...;
// R(z) ~ -3 / z), fifth order, |R(i w h)| on the stiff spring, the steps on
// the two masses. Both methods converge at their orders on the bushed
// pendulum with soft bushings (bushed_pendulum.hpp), a smooth spatial model;
// with stiff bushings SDIRK4 loses its order (stage order 1, D33) and Radau
// IIA, with error control, meets a tight error for fewer evaluations than
// generalised-alpha: the plan's "done when".

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/bushed_pendulum.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using Complex = std::complex<Real>;

namespace {

/// A linear spring on the coordinate of a slider (body's joint, at index i).
class Spring : public JointForce {
public:
    Spring(int body, Index i, Real k) : body_(body), i_(i), k_(k) {}
    const char* name() const override { return "spring"; }
    std::vector<int> bodies() const override { return {body_}; }
    void apply(const VecX& q, const VecX&, VecX& tau) const override { tau(i_) -= k_ * q(i_); }

private:
    int body_;
    Index i_;
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
        sys.joint_forces.push_back(std::make_shared<Spring>(1, 0, k));
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

/// Generalised-alpha's recurrence for q'' = -w^2 q, started from a = qdd.
struct AlphaRecurrence {
    Real W, h, am, af, g, b, q, v, a;

    AlphaRecurrence(Real w, Real h_, Real rho, Real q0, Real v0)
        : W(w * w), h(h_), am((2.0 * rho - 1.0) / (rho + 1.0)), af(rho / (rho + 1.0)),
          g(0.5 - am + af), b(0.25 * (g + 0.5) * (g + 0.5)), q(q0), v(v0), a(-W * q0)
    {}

    void step()
    {
        const Real qdd = -W * q;
        const Real a1 = (-(1.0 - af) * W * (q + h * v + h * h * (0.5 - b) * a) - am * a + af * qdd)
                        / ((1.0 - am) + (1.0 - af) * b * h * h * W);
        q += h * v + h * h * ((0.5 - b) * a + b * a1);
        v += h * ((1.0 - g) * a + g * a1);
        a = a1;
    }
};

/// SDIRK4's stability function, R(z) = 1 + z b^T (I - z A)^-1 1, by forward
/// substitution through its lower triangular A.
Complex sdirk_stability(Complex z)
{
    static constexpr Real A[5][5] = {{0.25, 0.0, 0.0, 0.0, 0.0},
                                     {0.5, 0.25, 0.0, 0.0, 0.0},
                                     {17.0 / 50.0, -1.0 / 25.0, 0.25, 0.0, 0.0},
                                     {371.0 / 1360.0, -137.0 / 2720.0, 15.0 / 544.0, 0.25, 0.0},
                                     {25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25}};
    Complex k[5];
    Complex r = 1.0;
    for (int i = 0; i < 5; ++i) {
        Complex s = 1.0;
        for (int j = 0; j < i; ++j) s += z * A[i][j] * k[j];
        k[i] = s / (1.0 - z * A[i][i]);
        r += z * A[4][i] * k[i];   // the weights are the last row
    }
    return r;
}

/// Radau IIA's stability function from its coefficients: the stages'
/// increments solve (I - z A) Z = z A 1, and the new value is 1 + Z_3
/// (stiffly accurate).
Complex radau_stability(Complex z)
{
    const Real s6 = std::sqrt(6.0);
    Eigen::Matrix3d A;
    A << (88.0 - 7.0 * s6) / 360.0, (296.0 - 169.0 * s6) / 1800.0, (-2.0 + 3.0 * s6) / 225.0,
        (296.0 + 169.0 * s6) / 1800.0, (88.0 + 7.0 * s6) / 360.0, (-2.0 - 3.0 * s6) / 225.0,
        (16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0;
    const Eigen::Matrix3cd M = Eigen::Matrix3cd::Identity() - z * A.cast<Complex>();
    const Eigen::Vector3cd Z = M.inverse() * (z * A.cast<Complex>() * Eigen::Vector3cd::Ones());
    return 1.0 + Z(2);
}

/// N steps of the recurrence; returns (q, v).
std::pair<Real, Real> alpha_recurrence(Real w, Real h, int N, Real rho, Real q, Real v)
{
    AlphaRecurrence r(w, h, rho, q, v);
    for (int k = 0; k < N; ++k) r.step();
    return {r.q, r.v};
}

/// Two 1 kg blocks along Z: the first on a slider to the ground with a soft
/// spring k_s, the second on a slider on the first with a stiff spring k_t.
/// In the sliders' coordinates M = [[2, 1], [1, 1]], K = diag(k_s, k_t).
struct TwoMasses {
    System sys;
    TwoMasses(Real k_s, Real k_t)
    {
        sys.model.gravity = Vec3::Zero();
        const auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
        const int b1 = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                          inertia, "first");
        const int b2 = sys.model.add_body(b1, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                          inertia, "second");
        sys.joint_forces.push_back(std::make_shared<Spring>(b1, 0, k_s));
        sys.joint_forces.push_back(std::make_shared<Spring>(b2, 1, k_t));
    }
};

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

TEST_CASE("Generalised-alpha: the steps are the method's recurrence, at second order",
          "[kernel][integrators][implicit]")
{
    // Against the recurrence: equal to rounding, Newton's tolerance being set
    // to 1e-13 here (with an iteration matrix exact for a linear system, two
    // iterations reach it). Against the exact solution the error is the
    // method's phase lag, 0.0880 (w h)^2 radians per radian for rho_infinity
    // = 0.8 (computed from the recurrence; 1/12 at rho_infinity = 1, the
    // trapezoidal rule's): 3.5e-5 m after 10 s at h = 0.02 s. Halving the
    // step divides it by 4.
    Oscillator osc;
    const Real T = 10.0;
    auto error_at = [&](Real h, IntegratorStats& stats) {
        auto sim = osc.simulator();
        sim->method = Integrator::GeneralizedAlpha;
        sim->step_control.relative_tolerance = 0.0;
        sim->step_control.absolute_tolerance = 1e-13;
        const int steps = sim->run(T, h);
        const auto [q, v] = alpha_recurrence(1.0, h, steps, 0.8, Oscillator::a, 0.0);
        INFO("h = " << h << ": q " << sim->q(0) << ", v " << sim->v(0) << "; recurrence " << q << ", " << v);
        CHECK(std::abs(sim->q(0) - q) < 1e-12);
        CHECK(std::abs(sim->v(0) - v) < 1e-12);
        stats = sim->integrator_stats();
        // One evaluation at each step's start and one per Newton iteration;
        // two more, once, for the iteration matrix (d/dq and d/dv).
        CHECK(stats.evaluations == steps + stats.newton_iterations + 2);
        return std::abs(w_of(*sim) - Oscillator::a * std::exp(Complex(0.0, T)));
    };
    IntegratorStats s1, s2;
    const Real e1 = error_at(0.02, s1), e2 = error_at(0.01, s2);
    const Real lag = Oscillator::a * 0.0880 * T * 0.02 * 0.02;
    INFO("errors " << e1 << " and " << e2 << ", predicted " << lag << " at h = 0.02 s; Newton iterations "
                   << s1.newton_iterations << " and " << s2.newton_iterations << " in " << s1.steps << " and "
                   << s2.steps << " steps; iteration matrices " << s1.jacobians << ", " << s2.jacobians);
    CHECK(std::abs(e1 / lag - 1.0) < 0.01);
    CHECK(e1 / e2 > 3.95);
    CHECK(e1 / e2 < 4.05);
    // The system is linear: one iteration matrix serves the whole run.
    CHECK(s1.jacobians == 1);
    CHECK(s2.jacobians == 1);
    CHECK(s1.newton_iterations <= 2 * s1.steps);
}

TEST_CASE("Generalised-alpha: a spring 1000 times too stiff for RK4 at 1 ms runs stably",
          "[kernel][integrators][implicit]")
{
    // RK4 at step h is stable on an undamped oscillator while w h <= 2
    // sqrt(2), since |R(iy)|^2 = 1 - y^6/72 + y^8/576: at 1 ms and 1 kg, for
    // springs up to 8e6 N/m. Just inside (w h = 0.95 x 2 sqrt(2)) the
    // amplitude never grows (|R| = 0.70); just outside (1.05 x, |R| = 1.40) it
    // passes 1000 times its start within 100 steps. A spring 1000 times
    // stiffer, 8e9 N/m (w h = 89.4): generalised-alpha stays bounded and, its
    // spectral radius there being near rho_infinity = 0.8, has removed the
    // motion within 1000 steps. On the way the amplitude first grows, to
    // about twice its start: at such a step the step matrix's eigenvalues lie
    // close together near -rho_infinity, and a matrix like that stretches a
    // vector before it shrinks it. The recurrence does the same, to the
    // digit. At rho_infinity = 1 (no damping) the energy is kept, to Newton's
    // tolerance.
    const Real h = 1e-3, w_limit = 2.0 * std::sqrt(2.0) / h;
    auto amplitude = [](const Simulator& s, Real w) { return std::hypot(s.q(0), s.v(0) / w); };
    {
        const Real w = 0.95 * w_limit;
        Oscillator osc(w * w);
        auto sim = osc.simulator();
        sim->method = Integrator::RK4;
        Real largest = 0.0;
        for (int k = 0; k < 1000; ++k) {
            sim->step(h);
            largest = std::max(largest, amplitude(*sim, w));
        }
        CHECK(largest <= Oscillator::a * (1.0 + 1e-12));
    }
    {
        const Real w = 1.05 * w_limit;
        Oscillator osc(w * w);
        auto sim = osc.simulator();
        sim->method = Integrator::RK4;
        sim->run(0.1, h);
        CHECK(amplitude(*sim, w) > 1e3 * Oscillator::a);
    }
    const Real w = std::sqrt(1000.0) * w_limit;
    for (const Real rho : {0.8, 1.0}) {
        Oscillator osc(w * w);
        auto sim = osc.simulator();
        sim->method = Integrator::GeneralizedAlpha;
        sim->implicit_control.rho_infinity = rho;
        sim->step_control.relative_tolerance = 1e-12;
        sim->step_control.absolute_tolerance = 1e-12;
        auto energy = [&] { return 0.5 * sim->v(0) * sim->v(0) + 0.5 * w * w * sim->q(0) * sim->q(0); };
        const Real E0 = energy();
        AlphaRecurrence recurrence(w, h, rho, Oscillator::a, 0.0);
        Real largest = 0.0, largest_recurrence = 0.0;
        for (int k = 0; k < 1000; ++k) {
            sim->step(h);
            recurrence.step();
            largest = std::max(largest, amplitude(*sim, w));
            largest_recurrence = std::max(largest_recurrence, std::hypot(recurrence.q, recurrence.v / w));
        }
        INFO("rho_infinity " << rho << ": largest amplitude " << largest << " m (recurrence "
                             << largest_recurrence << " m), at the end " << amplitude(*sim, w) << " m; energy "
                             << energy() / E0 << " of the start; " << sim->integrator_stats().newton_iterations
                             << " Newton iterations, " << sim->integrator_stats().jacobians
                             << " iteration matrices");
        CHECK(std::abs(largest / largest_recurrence - 1.0) < 1e-9);
        if (rho < 1.0) CHECK(amplitude(*sim, w) < 1e-6 * Oscillator::a);
        else CHECK(std::abs(energy() / E0 - 1.0) < 1e-8);
    }
}

TEST_CASE("Generalised-alpha: a stiff link leaves the slow motion second-order accurate",
          "[kernel][integrators][implicit]")
{
    // k_s = 2 (2 pi)^2 N/m gives a slow mode near 1 Hz; k_t = 4e9 N/m a
    // stiff one of w2 h = 89 at 1 ms, 1000 times beyond RK4's reach. The
    // modes: w^4 - (k_s + 2 k_t) w^2 + k_s k_t = 0, shapes (1, w^2 / (k_t -
    // w^2)). Released from q = (a, 0) at rest, the stiff mode's share is about
    // k_s / (4 k_t) a = 5e-9 a. The method is linear, and the same in every
    // mode, so the simulator must equal the recurrence run mode by mode; and
    // the slow motion must lag the exact one by the method's phase error,
    // 0.0880 (w1 h)^2 per radian: 4.4e-5 a after 2 s.
    const Real a = 0.01, h = 1e-3, T = 2.0;
    const Real k_s = 2.0 * std::pow(2.0 * pi, 2), k_t = 4e9;
    const Real B = k_s + 2.0 * k_t, W2 = 0.5 * (B + std::sqrt(B * B - 4.0 * k_s * k_t)), W1 = k_s * k_t / W2;
    const Real w1 = std::sqrt(W1), w2 = std::sqrt(W2);
    const Real r1 = W1 / (k_t - W1), r2 = W2 / (k_t - W2);
    const Real c1 = a * r2 / (r2 - r1), c2 = -a * r1 / (r2 - r1);

    TwoMasses tm(k_s, k_t);
    Simulator sim(tm.sys);
    sim.method = Integrator::GeneralizedAlpha;
    sim.step_control.relative_tolerance = 1e-12;
    sim.step_control.absolute_tolerance = 1e-14;
    sim.q(0) = a;
    const int steps = sim.run(T, h);

    const auto [slow, slow_rate] = alpha_recurrence(w1, h, steps, 0.8, c1, 0.0);
    const auto [fast, fast_rate] = alpha_recurrence(w2, h, steps, 0.8, c2, 0.0);
    INFO("w1 " << w1 << ", w2 " << w2 << " rad/s; stiff share " << c2 / a << " of a");
    CHECK(std::abs(sim.q(0) - (slow + fast)) < 1e-10 * a);
    CHECK(std::abs(sim.q(1) - (r1 * slow + r2 * fast)) < 1e-10 * a);
    CHECK(std::abs(sim.v(0) - (slow_rate + fast_rate)) < 1e-10 * a * w1);

    const Real error = std::abs(Complex(sim.q(0), -sim.v(0) / w1) - c1 * std::exp(Complex(0.0, w1 * T)));
    const Real lag = c1 * 0.0880 * w1 * T * (w1 * h) * (w1 * h);
    INFO("slow motion's error " << error / a << " a, predicted " << lag / a << " a; " << sim.integrator_stats().jacobians
                                << " iteration matrices, " << sim.integrator_stats().newton_iterations
                                << " Newton iterations in " << steps << " steps");
    CHECK(std::abs(error / lag - 1.0) < 0.05);
}

TEST_CASE("Generalised-alpha: bad settings and a step that cannot be solved are refused",
          "[kernel][integrators][implicit]")
{
    {
        Oscillator osc;
        auto sim = osc.simulator();
        sim->method = Integrator::GeneralizedAlpha;
        sim->implicit_control.rho_infinity = 1.5;
        CHECK_THROWS_WITH(sim->step(1e-3), ContainsSubstring("MBD-K059"));
    }
    {
        // A force that is not a number: the iteration matrix built from it
        // has no pivot.
        struct NotANumber : JointForce {
            const char* name() const override { return "not a number"; }
            std::vector<int> bodies() const override { return {1}; }
            void apply(const VecX&, const VecX&, VecX& tau) const override
            {
                tau(0) += std::numeric_limits<Real>::quiet_NaN();
            }
        };
        Oscillator osc;
        osc.sys.joint_forces.push_back(std::make_shared<NotANumber>());
        auto sim = osc.simulator();
        sim->method = Integrator::GeneralizedAlpha;
        CHECK_THROWS_WITH(sim->step(1e-3), ContainsSubstring("MBD-K058"));
        // SDIRK4 and Radau IIA at a fixed step refuse the same way; with error
        // control they would shorten the step until K055.
        for (const auto method : {Integrator::SDIRK4, Integrator::RadauIIA}) {
            auto other = osc.simulator();
            other->method = method;
            other->step_control.adaptive = false;
            CHECK_THROWS_WITH(other->step(1e-3), ContainsSubstring("MBD-K058"));
        }
    }
}

TEST_CASE("SDIRK4: one step is its stability function, and fixed steps converge at fourth order",
          "[kernel][integrators][implicit]")
{
    // One step of 0.3 s from (a, 0) is R(0.3 i) a, to Newton's tolerance (1e-14
    // here). Over T = 10 s each step loses |e^(ih) - R(ih)| = 8.45e-4 h^5 to
    // leading order, coherently: N |e^(ih) - R(ih)| a in all, 8.5e-8 m at h =
    // 0.1 s; halving the step divides it by 16.
    Oscillator osc;
    auto make = [&] {
        auto sim = osc.simulator();
        sim->method = Integrator::SDIRK4;
        sim->step_control.adaptive = false;
        sim->step_control.relative_tolerance = 0.0;
        sim->step_control.absolute_tolerance = 1e-14;
        return sim;
    };
    {
        auto sim = make();
        sim->step(0.3);
        const Complex expected = sdirk_stability(Complex(0.0, 0.3)) * Oscillator::a;
        INFO("w = " << w_of(*sim) << ", R(0.3 i) a = " << expected);
        CHECK(std::abs(w_of(*sim) - expected) < 1e-13);
    }
    const Real T = 10.0;
    auto error_at = [&](Real h, IntegratorStats& stats) {
        auto sim = make();
        const int steps = sim->run(T, h);
        const Complex R = sdirk_stability(Complex(0.0, h));
        INFO("h = " << h << ": w = " << w_of(*sim) << ", R(ih)^N a = " << std::pow(R, steps) * Oscillator::a);
        CHECK(std::abs(w_of(*sim) - std::pow(R, steps) * Oscillator::a) < 1e-12);
        stats = sim->integrator_stats();
        return std::abs(w_of(*sim) - Oscillator::a * std::exp(Complex(0.0, T)));
    };
    auto leading = [&](Real h) {
        const int steps = static_cast<int>(std::round(T / h));
        return steps * std::abs(std::exp(Complex(0.0, h)) - sdirk_stability(Complex(0.0, h))) * Oscillator::a;
    };
    IntegratorStats s1, s2;
    const Real e1 = error_at(0.1, s1), e2 = error_at(0.05, s2);
    INFO("errors " << e1 << " and " << e2 << ", predicted " << leading(0.1) << " and " << leading(0.05)
                   << "; Newton iterations " << s1.newton_iterations << " in " << s1.steps << " steps, "
                   << s1.jacobians << " iteration matrices, " << s1.evaluations << " evaluations");
    CHECK(std::abs(e1 / leading(0.1) - 1.0) < 0.01);
    CHECK(std::abs(e2 / leading(0.05) - 1.0) < 0.01);
    CHECK(e1 / e2 > 15.5);
    CHECK(e1 / e2 < 16.5);
    // Linear: one iteration matrix for the run, and every evaluation either a
    // Newton iteration or one of the three for that matrix (f, d/dq, d/dv).
    CHECK(s1.jacobians == 1);
    CHECK(s1.evaluations == s1.newton_iterations + 3);
}

TEST_CASE("SDIRK4: L-stable, a spring far too stiff for the step shrinks by |R(i w h)| in one step",
          "[kernel][integrators][implicit]")
{
    // The spring 1000 times too stiff for RK4 at 1 ms (w h = 89.4): one step
    // multiplies the state by R(89.4 i), |R| = 0.104 (L-stability: R(iy) ~ 9.3
    // / y for large y), where generalised-alpha keeps about rho_infinity = 0.8.
    const Real h = 1e-3, w = std::sqrt(1000.0) * 2.0 * std::sqrt(2.0) / h;
    Oscillator osc(w * w);
    auto sim = osc.simulator();
    sim->method = Integrator::SDIRK4;
    sim->step_control.adaptive = false;
    sim->step_control.relative_tolerance = 1e-12;
    sim->step_control.absolute_tolerance = 1e-12;
    sim->step(h);
    const Complex R = sdirk_stability(Complex(0.0, w * h));
    const Complex w_sim(sim->q(0), -sim->v(0) / w);
    INFO("state " << w_sim << ", R(i w h) a = " << R * Oscillator::a << ", |R| = " << std::abs(R));
    CHECK(std::abs(w_sim - R * Oscillator::a) < 1e-9 * Oscillator::a);
    CHECK(std::abs(R) < 0.11);
}

TEST_CASE("SDIRK4: with error control on a stiff system the steps follow the slow motion",
          "[kernel][integrators][implicit]")
{
    // The two masses with the stiff link (w2 = 8.9e4 rad/s), released in the
    // slow mode's shape, q = a (1, r1), so that the stiff mode, undamped, is
    // not excited: released from (a, 0) it would carry a velocity of 4.5e-6
    // m/s for ever, part of the exact solution, which an error control at
    // 1e-8 must resolve (and does, at steps of its period). The error
    // estimate goes as (b - b_hat)^T A^3 1 z^4 = -0.00911 (w1 h)^4 on the slow
    // motion, so the steps are set by the slow motion, not by the stiffness
    // in the iteration matrix: at tolerance 1e-8 (absolute) the measure
    // 0.00911 (w1 h)^4 c1 sqrt((1 + w1^2) / 2) / (2 tol) held near 0.9^4 gives
    // w1 h = 0.075, h = 12 ms, some 170 steps over 2 s, against a stiff period
    // of 7e-5 s. The fourth-order solution's own error per step is smaller
    // than the estimate by about (8.45e-4 / 0.00911) w1 h = 0.007, so over 170
    // steps the slow motion's error stays near tol, 1e-8 m.
    const Real a = 0.01, T = 2.0, k_s = 2.0 * std::pow(2.0 * pi, 2), k_t = 4e9;
    const Real B = k_s + 2.0 * k_t, W2 = 0.5 * (B + std::sqrt(B * B - 4.0 * k_s * k_t)), W1 = k_s * k_t / W2;
    const Real w1 = std::sqrt(W1);
    const Real r1 = W1 / (k_t - W1);
    const Real c1 = a;
    TwoMasses tm(k_s, k_t);
    Simulator sim(tm.sys);
    sim.method = Integrator::SDIRK4;
    sim.step_control.relative_tolerance = 0.0;
    sim.step_control.absolute_tolerance = 1e-8;
    sim.q(0) = a;
    sim.q(1) = r1 * a;
    sim.step(T);
    const auto& stats = sim.integrator_stats();
    const Real error = std::abs(Complex(sim.q(0), -sim.v(0) / w1) - c1 * std::exp(Complex(0.0, w1 * T)));
    INFO("slow motion's error " << error << " m; " << stats.steps << " steps, " << stats.rejected << " rejected, "
                                << stats.newton_iterations << " Newton iterations, " << stats.jacobians
                                << " iteration matrices");
    CHECK(stats.steps > 100);
    CHECK(stats.steps < 300);
    CHECK(error < 3e-8);
}

TEST_CASE("Radau IIA: one step is its stability function, and fixed steps converge at fifth order",
          "[kernel][integrators][implicit]")
{
    // One step of 0.3 s from (a, 0) is R(0.3 i) a, to Newton's tolerance (1e-14
    // here). Over T = 10 s each step loses |e^(ih) - R(ih)| = h^6 / 7200 to
    // leading order, coherently: N |e^(ih) - R(ih)| a in all, 1.4e-9 m at h =
    // 0.1 s; halving the step divides it by 32.
    Oscillator osc;
    auto make = [&] {
        auto sim = osc.simulator();
        sim->method = Integrator::RadauIIA;
        sim->step_control.adaptive = false;
        sim->step_control.relative_tolerance = 0.0;
        sim->step_control.absolute_tolerance = 1e-14;
        return sim;
    };
    {
        auto sim = make();
        sim->step(0.3);
        const Complex expected = radau_stability(Complex(0.0, 0.3)) * Oscillator::a;
        INFO("w = " << w_of(*sim) << ", R(0.3 i) a = " << expected);
        CHECK(std::abs(w_of(*sim) - expected) < 1e-13);
    }
    const Real T = 10.0;
    auto error_at = [&](Real h, IntegratorStats& stats) {
        auto sim = make();
        const int steps = sim->run(T, h);
        const Complex R = radau_stability(Complex(0.0, h));
        INFO("h = " << h << ": w = " << w_of(*sim) << ", R(ih)^N a = " << std::pow(R, steps) * Oscillator::a);
        CHECK(std::abs(w_of(*sim) - std::pow(R, steps) * Oscillator::a) < 1e-12);
        stats = sim->integrator_stats();
        return std::abs(w_of(*sim) - Oscillator::a * std::exp(Complex(0.0, T)));
    };
    auto leading = [&](Real h) {
        const int steps = static_cast<int>(std::round(T / h));
        return steps * std::abs(std::exp(Complex(0.0, h)) - radau_stability(Complex(0.0, h))) * Oscillator::a;
    };
    IntegratorStats s1, s2;
    const Real e1 = error_at(0.2, s1), e2 = error_at(0.1, s2);
    INFO("errors " << e1 << " and " << e2 << ", predicted " << leading(0.2) << " and " << leading(0.1)
                   << "; Newton iterations " << s1.newton_iterations << " in " << s1.steps << " steps, "
                   << s1.jacobians << " iteration matrices, " << s1.evaluations << " evaluations");
    CHECK(std::abs(e1 / leading(0.2) - 1.0) < 0.01);
    CHECK(std::abs(e2 / leading(0.1) - 1.0) < 0.01);
    CHECK(e1 / e2 > 31.0);
    CHECK(e1 / e2 < 33.0);
    // Linear: one iteration matrix for the run; three evaluations per Newton
    // iteration and three for that matrix (f, d/dq, d/dv); none for an error
    // estimate, which fixed steps do not make.
    CHECK(s1.jacobians == 1);
    CHECK(s1.evaluations == 3 * s1.newton_iterations + 3);
}

TEST_CASE("Radau IIA: L-stable, a spring far too stiff for the step shrinks by |R(i w h)| in one step",
          "[kernel][integrators][implicit]")
{
    // The spring 1000 times too stiff for RK4 at 1 ms (w h = 89.4): one step
    // multiplies the state by R(89.4 i), |R| = 0.0336 (L-stability: R(z) ~ -3
    // / z for large z), against SDIRK4's 0.104 and generalised-alpha's 0.8.
    const Real h = 1e-3, w = std::sqrt(1000.0) * 2.0 * std::sqrt(2.0) / h;
    Oscillator osc(w * w);
    auto sim = osc.simulator();
    sim->method = Integrator::RadauIIA;
    sim->step_control.adaptive = false;
    sim->step_control.relative_tolerance = 1e-12;
    sim->step_control.absolute_tolerance = 1e-12;
    sim->step(h);
    const Complex R = radau_stability(Complex(0.0, w * h));
    const Complex w_sim(sim->q(0), -sim->v(0) / w);
    INFO("state " << w_sim << ", R(i w h) a = " << R * Oscillator::a << ", |R| = " << std::abs(R));
    CHECK(std::abs(w_sim - R * Oscillator::a) < 1e-9 * Oscillator::a);
    CHECK(std::abs(R) < 0.035);
}

TEST_CASE("Radau IIA: with error control on a stiff system the steps follow the slow motion",
          "[kernel][integrators][implicit]")
{
    // As for SDIRK4: the two masses with the stiff link, released in the slow
    // mode's shape. RADAU5's estimate, (gamma - z)^-1 (z + dd^T Z(z)) for y'
    // = lambda y, goes as -0.00458 z^4 (computed from the coefficients; half
    // SDIRK4's), so at tolerance 1e-8 the steps settle where SDIRK4's measure
    // gives w1 h = 0.075 times 2^(1/4): w1 h = 0.089, h = 14 ms, some 140 steps
    // over 2 s. The fifth-order solution's own error is then (w1 h)^6 / 7200
    // = 7e-11 of the amplitude per step, about 1e-10 m over the run.
    const Real a = 0.01, T = 2.0, k_s = 2.0 * std::pow(2.0 * pi, 2), k_t = 4e9;
    const Real B = k_s + 2.0 * k_t, W2 = 0.5 * (B + std::sqrt(B * B - 4.0 * k_s * k_t)), W1 = k_s * k_t / W2;
    const Real w1 = std::sqrt(W1);
    const Real r1 = W1 / (k_t - W1);
    TwoMasses tm(k_s, k_t);
    Simulator sim(tm.sys);
    sim.method = Integrator::RadauIIA;
    sim.step_control.relative_tolerance = 0.0;
    sim.step_control.absolute_tolerance = 1e-8;
    sim.q(0) = a;
    sim.q(1) = r1 * a;
    sim.step(T);
    const auto& stats = sim.integrator_stats();
    const Real error = std::abs(Complex(sim.q(0), -sim.v(0) / w1) - a * std::exp(Complex(0.0, w1 * T)));
    INFO("slow motion's error " << error << " m; " << stats.steps << " steps, " << stats.rejected << " rejected, "
                                << stats.newton_iterations << " Newton iterations, " << stats.jacobians
                                << " iteration matrices, " << stats.evaluations << " evaluations");
    CHECK(stats.steps > 90);
    CHECK(stats.steps < 250);
    CHECK(error < 1e-9);
}

TEST_CASE("SDIRK4 and Radau IIA converge at their orders on a smooth spatial model",
          "[kernel][integrators][implicit]")
{
    // The bushed pendulum with soft bushings (k = 1e4 N/m, k_r = 1e2 N m/rad:
    // its fastest motion near 140 rad/s, resolved by every step here), so
    // nothing is stiff: two free bodies turning through large angles, their
    // quaternions carried as four numbers through the stages. Halving the
    // step must divide the tip's error after 1 s by 2^4 = 16 for SDIRK4 and
    // 2^5 = 32 for Radau IIA, to within the next term (a few per cent at
    // these steps). The reference: Dormand-Prince at tolerance 1e-12.
    mbd_test::BushedPendulum bp(1e4, 1e2);
    auto tip_after = [&](Integrator method, Real h, Real tol, bool adaptive) {
        Simulator sim(bp.sys);
        sim.method = method;
        bp.release(sim);
        sim.step_control.adaptive = adaptive;
        sim.step_control.relative_tolerance = tol;
        sim.step_control.absolute_tolerance = tol;
        sim.run(1.0, h);
        return bp.tip(sim);
    };
    const Vec3 reference = tip_after(Integrator::DormandPrince, 0.01, 1e-12, true);
    auto error = [&](Integrator method, Real h) { return (tip_after(method, h, 1e-10, false) - reference).norm(); };
    const Real s1 = error(Integrator::SDIRK4, 5e-3), s2 = error(Integrator::SDIRK4, 2.5e-3);
    const Real r1 = error(Integrator::RadauIIA, 1e-2), r2 = error(Integrator::RadauIIA, 5e-3);
    INFO("SDIRK4 at 5 and 2.5 ms: " << s1 << ", " << s2 << " (ratio " << s1 / s2 << "); Radau IIA at 10 and 5 ms: "
                                    << r1 << ", " << r2 << " (ratio " << r1 / r2 << ")");
    CHECK(s1 / s2 > 15.0);
    CHECK(s1 / s2 < 17.0);
    CHECK(r1 / r2 > 30.0);
    CHECK(r1 / r2 < 34.0);
}

TEST_CASE("Radau IIA beats generalised-alpha on the bushed pendulum at tight tolerance",
          "[kernel][integrators][implicit][benchmark]")
{
    // The bushed benchmark (bushed_pendulum.hpp, k = 1e8 N/m, k_r = 1e6 N m/
    // rad), 1 s from release, and the plan's "done when" for task 4.4: the
    // tip within 1e-6 m (on a swing of a metre) for fewer evaluations than
    // generalised-alpha needs. The reference: Dormand-Prince at tolerance
    // 1e-12, an explicit method independent of both (the bushings hold its
    // steps near its stability limit, which costs steps, not accuracy).
    // Generalised-alpha has no error control: its fixed steps are halved
    // until it meets the target (second order: 1.7e-6 m at 0.5 ms, 4.6e-7 at
    // 0.25 ms when measured), with Newton at 1e-9, tight enough over 4000
    // steps. Radau IIA runs as it would be used, with error control at the
    // target. SDIRK4, whose stage order is 1, cannot compete here (D33): at
    // 0.5 ms it is still 4e-5 m out.
    mbd_test::BushedPendulum bp;
    const Real T = 1.0, target = 1e-6;
    auto run = [&](Integrator method, Real h, Real tol, bool adaptive, long long& evaluations) {
        Simulator sim(bp.sys);
        sim.method = method;
        bp.release(sim);
        sim.step_control.adaptive = adaptive;
        sim.step_control.relative_tolerance = tol;
        sim.step_control.absolute_tolerance = tol;
        sim.run(T, h);
        evaluations = sim.integrator_stats().evaluations;
        return bp.tip(sim);
    };
    long long reference_cost = 0, alpha_cost = -1, radau_cost = 0;
    const Vec3 reference = run(Integrator::DormandPrince, 0.01, 1e-12, true, reference_cost);
    std::ostringstream table;
    table << "reference: " << reference_cost << " evaluations\n";
    for (const Real h : {5e-4, 2.5e-4, 1.25e-4}) {
        long long n = 0;
        const Real e = (run(Integrator::GeneralizedAlpha, h, 1e-9, false, n) - reference).norm();
        table << "generalised-alpha at " << h << " s: error " << e << " m, " << n << " evaluations\n";
        if (e <= target) {
            alpha_cost = n;
            break;
        }
    }
    const Real radau_error = (run(Integrator::RadauIIA, 0.01, target, true, radau_cost) - reference).norm();
    table << "Radau IIA at tolerance " << target << ": error " << radau_error << " m, " << radau_cost << " evaluations";
    INFO(table.str());
    REQUIRE(alpha_cost > 0);
    CHECK(radau_error <= target);
    CHECK(radau_cost < alpha_cost);
}

namespace {

/// The generalized forces of a system's force elements and joint forces at
/// (q, v), from the kernel's functions directly: what force_jacobian
/// differentiates.
VecX applied_forces(const System& sys, const VecX& q, const VecX& v)
{
    Data data(sys.model);
    forward_kinematics(sys.model, data, q, v);
    std::vector<RigidBodyState> states;
    body_states(sys.model, data, states);
    std::vector<RigidBodyForces> forces(states.size());
    for (const auto& element : sys.force_elements) element->apply(states, forces);
    VecX tau;
    generalized_forces(sys.model, data, forces, tau);
    for (const auto& jf : sys.joint_forces) jf->apply(q, v, tau);
    return tau;
}

/// Warnings caught while a test replaces the diagnostic sink.
std::vector<std::string>& caught()
{
    static std::vector<std::string> messages;
    return messages;
}

} // namespace

TEST_CASE("Linearly implicit Euler: on a linear system each step is backward Euler's",
          "[kernel][integrators][implicit][linear]")
{
    // A 1 kg block on a spring (k = 1 N/m) and a damper (c = 0.3 N s/m),
    // both joint forces. Backward Euler, q+ = q + h v+, v+ = v + h (-k q+ - c
    // v+), gives (1 + h c + h^2 k) v+ = v - h k q; on a linear system the
    // linearly implicit step is that exactly, its derivatives being exact up
    // to the differences' rounding (about 1e-10 of k and c). 100 steps of 0.1
    // s; one evaluation and one set of derivatives per step.
    const Real k = 1.0, c = 0.3, h = 0.1;
    Oscillator osc(k);
    JointCoordinateForceParams damper;
    damper.damper = Curve::linear(c);
    osc.sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(osc.sys.model, 1, damper));
    auto sim = osc.simulator();
    sim->method = Integrator::LinearlyImplicitEuler;
    Real q = Oscillator::a, v = 0.0, largest = 0.0;
    for (int n = 0; n < 100; ++n) {
        sim->step(h);
        v = (v - h * k * q) / (1.0 + h * c + h * h * k);
        q += h * v;
        largest = std::max({largest, std::abs(sim->q(0) - q), std::abs(sim->v(0) - v)});
    }
    const auto& stats = sim->integrator_stats();
    INFO("largest difference from the recurrence " << largest << "; " << stats.steps << " steps, "
                                                    << stats.evaluations << " evaluations, " << stats.jacobians
                                                    << " sets of derivatives");
    CHECK(largest < 1e-10);
    CHECK(stats.evaluations == 100);
    CHECK(stats.jacobians == 100);
}

TEST_CASE("Linearly implicit Euler: a spring far too stiff for the step shrinks as backward Euler says",
          "[kernel][integrators][implicit][linear]")
{
    // The spring 1000 times too stiff for RK4 at 1 ms (w h = 89.4). Backward
    // Euler's step on q'' = -w^2 q multiplies the state by 1 / (1 -+ i w h), of
    // modulus 1 / sqrt(1 + (w h)^2) = 0.0112: one step from (a, 0) gives v =
    // -h w^2 a / (1 + (w h)^2), q = a + h v.
    const Real h = 1e-3, w = std::sqrt(1000.0) * 2.0 * std::sqrt(2.0) / h, W = w * w;
    Oscillator osc(W);
    auto sim = osc.simulator();
    sim->method = Integrator::LinearlyImplicitEuler;
    sim->step(h);
    const Real v = -h * W * Oscillator::a / (1.0 + h * h * W), q = Oscillator::a + h * v;
    const Real amplitude = std::abs(Complex(sim->q(0), -sim->v(0) / w));
    INFO("q " << sim->q(0) << " (" << q << "), v " << sim->v(0) << " (" << v << "); amplitude ratio "
              << amplitude / Oscillator::a);
    CHECK(std::abs(sim->q(0) - q) < 1e-9 * Oscillator::a);
    CHECK(std::abs(sim->v(0) - v) < 1e-9 * w * Oscillator::a);
    CHECK(std::abs(amplitude / Oscillator::a - 1.0 / std::sqrt(1.0 + h * h * W)) < 1e-9);
}

TEST_CASE("Linearly implicit Euler: first order on a smooth spatial model",
          "[kernel][integrators][implicit][linear]")
{
    // The bushed pendulum with soft bushings (as for SDIRK4 and Radau IIA
    // above): two free bodies turning through large angles. A first-order
    // method's error after 1 s halves when the step does: the ratio within
    // 10 % of 2 at 2 and 1 ms. The reference: Dormand-Prince at 1e-12.
    mbd_test::BushedPendulum bp(1e4, 1e2);
    auto tip_after = [&](Integrator method, Real h) {
        Simulator sim(bp.sys);
        sim.method = method;
        bp.release(sim);
        sim.step_control.relative_tolerance = 1e-12;
        sim.step_control.absolute_tolerance = 1e-12;
        sim.run(1.0, h);
        return bp.tip(sim);
    };
    const Vec3 reference = tip_after(Integrator::DormandPrince, 0.01);
    const Real e1 = (tip_after(Integrator::LinearlyImplicitEuler, 2e-3) - reference).norm();
    const Real e2 = (tip_after(Integrator::LinearlyImplicitEuler, 1e-3) - reference).norm();
    INFO("errors " << e1 << " and " << e2 << " m at 2 and 1 ms (ratio " << e1 / e2 << ")");
    CHECK(e1 / e2 > 1.8);
    CHECK(e1 / e2 < 2.2);
}

TEST_CASE("force_jacobian: element by element, the derivatives of the generalized forces",
          "[kernel][integrators][implicit][linear]")
{
    // The bushed pendulum (two free bodies, two bushings) with a spring-damper
    // between points off the links' axes as well. C = d tau / dv: element by
    // element it must equal central differences of the generalized forces
    // computed by the kernel's own functions, at any state; here a deflected,
    // moving one. K = d tau / dq: element by element it leaves out the
    // joints' geometry (how a load maps to generalized forces as the bodies
    // turn), which multiplies the loads, so it must equal the differences
    // where every load is zero: undeflected, at rest. The elements'
    // derivatives are forward differences of 1e-7, the reference central ones
    // of 1e-6: both within about 1e-6 of the largest entry.
    mbd_test::BushedPendulum bp(1e7, 1e5);
    Simulator probe(bp.sys);
    bp.release(probe);
    const Vec3 a1(0.5, 0.0, 0.03), a2(0.4, 0.02, 0.0);
    const Real free_length = ((probe.states()[2].p_WB + probe.states()[2].q_WB * a2)
                              - (probe.states()[1].p_WB + probe.states()[1].q_WB * a1))
                                 .norm();
    SpringDamperParams sp;
    sp.free_length = free_length;
    sp.spring = Curve::linear(2e4);
    sp.damper = Curve::linear(300.0);
    bp.sys.force_elements.push_back(std::make_shared<SpringDamper>(bp.link1, bp.link2, a1, a2, sp));

    auto differences = [&](const VecX& q, const VecX& v, MatX& K, MatX& C) {
        const Index nv = bp.sys.model.nv;
        K.resize(nv, nv);
        C.resize(nv, nv);
        VecX dq = VecX::Zero(nv), qp(q.size()), qm(q.size());
        for (Index j = 0; j < nv; ++j) {
            const Real e = 1e-6;
            dq.setZero();
            dq(j) = e;
            integrate(bp.sys.model, q, dq, 1.0, qp);
            integrate(bp.sys.model, q, dq, -1.0, qm);
            K.col(j) = (applied_forces(bp.sys, qp, v) - applied_forces(bp.sys, qm, v)) / (2.0 * e);
            VecX vp = v, vm = v;
            vp(j) += e;
            vm(j) -= e;
            C.col(j) = (applied_forces(bp.sys, q, vp) - applied_forces(bp.sys, q, vm)) / (2.0 * e);
        }
    };
    MatX K, C, K_ref, C_ref;
    {
        // Deflected and moving: the damping.
        Simulator sim(bp.sys);
        bp.release(sim);
        sim.q(0) += 2e-4;
        sim.q(9) -= 1e-4;
        sim.v << 0.3, -0.2, 0.5, 0.1, 0.4, -0.3, 0.2, 0.6, -0.1, -0.5, 0.2, 0.3;
        sim.force_jacobian(K, C);
        differences(sim.q, sim.v, K_ref, C_ref);
        const Real scale = C_ref.cwiseAbs().maxCoeff();
        INFO("damping: largest entry " << scale << ", largest difference " << (C - C_ref).cwiseAbs().maxCoeff());
        CHECK((C - C_ref).cwiseAbs().maxCoeff() < 1e-6 * scale);
    }
    {
        // Undeflected and at rest: the stiffness.
        Simulator sim(bp.sys);
        bp.release(sim);
        sim.force_jacobian(K, C);
        differences(sim.q, sim.v, K_ref, C_ref);
        const Real scale = K_ref.cwiseAbs().maxCoeff();
        INFO("stiffness: largest entry " << scale << ", largest difference " << (K - K_ref).cwiseAbs().maxCoeff());
        CHECK((K - K_ref).cwiseAbs().maxCoeff() < 1e-6 * scale);
    }
}

TEST_CASE("Linearly implicit Euler: a matrix that is not positive definite is reported, and the step goes on",
          "[kernel][integrators][implicit][linear]")
{
    // A spring of negative stiffness (-4e6 N/m on 1 kg): at h = 1 ms, M - h^2 K
    // = 1 - 4 < 0, so the step drops the stiffness (MBD-K045, once) and is
    // then explicit in it; the steps still complete, finite.
    Oscillator osc(-4e6);
    auto sim = osc.simulator();
    sim->method = Integrator::LinearlyImplicitEuler;
    caught().clear();
    const DiagnosticSink saved = diagnostic_sink();
    diagnostic_sink() = [](const std::string& m) { caught().push_back(m); };
    sim->step(1e-3);
    sim->step(1e-3);
    diagnostic_sink() = saved;
    REQUIRE(caught().size() == 1);
    CHECK(caught()[0].find("MBD-K045") != std::string::npos);
    CHECK(std::isfinite(sim->q(0)));
}
