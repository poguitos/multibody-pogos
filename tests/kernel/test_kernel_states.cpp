// State elements (plan task 4.1): first-order states owned by elements,
// integrated with the multibody state.
//
//   - a lag state whose value pushes a mass: the integrators reach their
//     orders (4 for RK4, 1 for semi-implicit Euler) against the closed form,
//     which they can only if the force sees each stage's value of the state;
//   - the hooks run in their order: begin_evaluation before the forces, rates
//     after, once per evaluation; end_step once per step, and its changes
//     stay;
//   - an element attached twice is refused;
//   - multirate RK4 (task 4.6): with six fast steps of the lag per step of
//     the mass the step keeps RK4's fourth order, the lag not depending on
//     the mass; a multirate setting below 1, or with another method, is
//     refused.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <memory>
#include <string>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

/// A force that decays as f' = -f / tau from f0: an actuator's lag. Its value
/// at each evaluation is kept for the force callback to apply.
class Lag : public StateElement {
public:
    Lag(Real f0, Real tau) : f0_(f0), tau_(tau) {}
    const char* name() const override { return "lag"; }
    int size() const override { return 1; }
    void initial_states(VecRef x) const override { x(0) = f0_; }
    void begin_evaluation(const Simulator&, Real, ConstVecRef x) override { force = x(0); }
    void rates(const Simulator&, Real, ConstVecRef x, VecRef x_dot) override { x_dot(0) = -x(0) / tau_; }

    Real force{0.0};

private:
    Real f0_, tau_;
};

/// A 2 kg block on a slider along Z, no gravity, pushed by a lag of 3 N
/// decaying with a time constant of 0.2 s.
struct PushedBlock {
    static constexpr Real m = 2.0, f0 = 3.0, tau = 0.2;
    System sys;
    Lag lag{f0, tau};

    PushedBlock()
    {
        sys.model.gravity = Vec3::Zero();
        sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                           RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.1)), "block");
    }

    /// The block's position at T, stepped by dt with the given method (and
    /// multirate setting); its statistics in `stats` if given.
    Real position(Integrator method, Real dt, Real T, int multirate = 1, IntegratorStats* stats = nullptr)
    {
        Simulator sim(sys);
        sim.method = method;
        sim.multirate = multirate;
        sim.attach(lag);
        sim.force_callback = [this](Simulator&, Real, VecX& tau_) { tau_(0) += lag.force; };
        sim.run(T, dt);
        if (stats) *stats = sim.integrator_stats();
        return sim.q(0);
    }

    /// The closed form: v = (f0 tau / m)(1 - e^(-t/tau)), x its integral.
    static Real exact(Real t) { return f0 * tau / m * (t - tau * (1.0 - std::exp(-t / tau))); }
};

} // namespace

TEST_CASE("State elements: a lag pushing a mass converges at the integrator's order", "[kernel][states]")
{
    // RK4 halving the step divides the error by 16 (between 14 and 18 allows
    // for the next term), semi-implicit Euler by 2 (1.8 to 2.2). With the
    // force taken from the state at the start of each step instead of each
    // stage, RK4 would fall to first order.
    PushedBlock pb;
    const Real T = 1.0, x_T = PushedBlock::exact(T);
    const Real e_rk4 = std::abs(pb.position(Integrator::RK4, 0.02, T) - x_T);
    const Real e_rk4_half = std::abs(pb.position(Integrator::RK4, 0.01, T) - x_T);
    INFO("RK4 errors " << e_rk4 << ", " << e_rk4_half);
    // About T lambda (lambda h)^4 / 120 of the motion's scale f0 tau / m =
    // 0.3 m, with lambda = 1 / tau = 5 per second and h = 0.02 s: 1.3e-6.
    CHECK(e_rk4 < 1e-5);
    CHECK(e_rk4 / e_rk4_half > 14.0);
    CHECK(e_rk4 / e_rk4_half < 18.0);

    const Real e_se = std::abs(pb.position(Integrator::SemiImplicitEuler, 1e-3, T) - x_T);
    const Real e_se_half = std::abs(pb.position(Integrator::SemiImplicitEuler, 5e-4, T) - x_T);
    INFO("semi-implicit Euler errors " << e_se << ", " << e_se_half);
    CHECK(e_se / e_se_half > 1.8);
    CHECK(e_se / e_se_half < 2.2);
}

TEST_CASE("State elements: the hooks run in order, and end_step changes stay", "[kernel][states]")
{
    // A recorder element: it logs each hook, and its end_step sets its state
    // to the number of steps so far, so that the next step starts from it.
    struct Recorder : StateElement {
        std::string log;
        int steps = 0;
        const char* name() const override { return "recorder"; }
        int size() const override { return 1; }
        void initial_states(VecRef x) const override { x(0) = 0.0; }
        void begin_evaluation(const Simulator&, Real, ConstVecRef) override { log += 'b'; }
        void rates(const Simulator&, Real, ConstVecRef, VecRef x_dot) override
        {
            log += 'r';
            x_dot(0) = 0.0;
        }
        void end_step(const Simulator&, Real, VecRef x) override
        {
            log += 'e';
            x(0) = ++steps;
        }
    };
    PushedBlock pb;
    Recorder rec;
    Simulator sim(pb.sys);
    sim.attach(rec);
    sim.force_callback = [&](Simulator&, Real, VecX&) { rec.log += 'f'; };
    sim.run(0.02, 0.01);
    // RK4: four evaluations per step, each begin, force, rates; then end_step.
    CHECK(rec.log == "bfrbfrbfrbfre" "bfrbfrbfrbfre");
    CHECK(sim.x(0) == 2.0);
    CHECK(sim.state_offset(0) == 0);
}

TEST_CASE("State elements: an element attached twice is refused", "[kernel][states]")
{
    PushedBlock pb;
    Simulator sim(pb.sys);
    sim.attach(pb.lag);
    CHECK_THROWS_WITH(sim.attach(pb.lag), ContainsSubstring("MBD-K044"));
    CHECK(sim.x.size() == 1);
}

TEST_CASE("Multirate RK4: a lag driving a mass keeps RK4's fourth order", "[kernel][states][multirate]")
{
    // Six fast steps of the lag per step of the block (task 4.6). The lag does
    // not depend on the block, so the block's stages see it at exactly their
    // times, from steps six times finer, and the step stays fourth order:
    // halving H divides the error by 16 (14 to 18 for the next term). RK4 on
    // the block's x'' = f(t) / m with f exact at its stages is Simpson's rule
    // for the velocity: an error of H^4 / 2880 times the integral of f'''' / m
    // = (f0 / m) e^(-t / tau) / tau^4, 6.5e-2 H^4 m/s after 1 s, 4.0e-7 m/s at
    // H = 0.05 s, and the position's of the same order: under 2e-6 m. (H
    // divides T: the run must end at T.) Each
    // step evaluates the dynamics four times and the lag's rate alone 23
    // times (four per fast step, the first given by the step's first
    // evaluation).
    PushedBlock pb;
    const Real T = 1.0, x_T = PushedBlock::exact(T);
    IntegratorStats s1, s2;
    const Real e1 = std::abs(pb.position(Integrator::RK4, 0.05, T, 6, &s1) - x_T);
    const Real e2 = std::abs(pb.position(Integrator::RK4, 0.025, T, 6, &s2) - x_T);
    INFO("errors " << e1 << " and " << e2 << " (ratio " << e1 / e2 << "); " << s1.steps << " steps, "
                   << s1.evaluations << " evaluations, " << s1.state_evaluations << " state evaluations");
    CHECK(e1 < 2e-6);
    CHECK(e1 / e2 > 14.0);
    CHECK(e1 / e2 < 18.0);
    CHECK(s1.evaluations == 4 * s1.steps);
    CHECK(s1.state_evaluations == 23 * s1.steps);
}

TEST_CASE("Multirate: a value below 1, or above 1 with a method other than RK4, is refused",
          "[kernel][states][multirate]")
{
    PushedBlock pb;
    Simulator sim(pb.sys);
    sim.attach(pb.lag);
    sim.multirate = 0;
    CHECK_THROWS_WITH(sim.step(0.01), ContainsSubstring("MBD-K046"));
    sim.multirate = 6;
    sim.method = Integrator::GeneralizedAlpha;
    CHECK_THROWS_WITH(sim.step(0.01), ContainsSubstring("MBD-K046"));
    sim.method = Integrator::RK4;
    CHECK_NOTHROW(sim.step(0.01));
}
