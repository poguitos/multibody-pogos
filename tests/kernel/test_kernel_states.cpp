// State elements (plan task 4.1): first-order states owned by elements,
// integrated with the multibody state.
//
//   - a lag state whose value pushes a mass: the integrators reach their
//     orders (4 for RK4, 1 for semi-implicit Euler) against the closed form,
//     which they can only if the force sees each stage's value of the state;
//   - the hooks run in their order: begin_evaluation before the forces, rates
//     after, once per evaluation; end_step once per step, and its changes
//     stay;
//   - an element attached twice is refused.

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

    /// The block's position at T, stepped by dt with the given method.
    Real position(Integrator method, Real dt, Real T)
    {
        Simulator sim(sys);
        sim.method = method;
        sim.attach(lag);
        sim.force_callback = [this](Simulator&, Real, VecX& tau_) { tau_(0) += lag.force; };
        sim.run(T, dt);
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
