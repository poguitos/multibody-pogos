// The kernel's force bridge and simulator (plan task 2.7).
//
//   - body forces become generalized forces as J^T f;
//   - a mass on a spring oscillates at sqrt(k / m), and the two integrators
//     converge at their orders (4 and 1);
//   - a constrained pendulum keeps its energy and its length;
//   - applied forces and callbacks act when and as often as documented.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::KJoint;
using mbd_test::Rng;

namespace {

/// One body on a free joint (frames: world and body), COM at its origin.
System free_body_system(Real mass, const Vec3& gravity)
{
    System sys;
    sys.model.gravity = gravity;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(mass, Vec3(0.1, 0.1, 0.1)));
    return sys;
}

void place(kernel::Simulator& sim, const Vec3& p)
{
    sim.q.head<3>() = p;
    sim.q.tail<4>() = Quat::Identity().coeffs();
}

/// x(T) of a mass m on a spring k from x0 = rest + 0.2, at rest, stepped by dt.
Real spring_position(Integrator method, Real dt, Real T)
{
    System sys = free_body_system(2.0, Vec3::Zero());
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, 1, Vec3::Zero(), Vec3::Zero(), 50.0, 0.0, 1.0));
    kernel::Simulator sim(sys);
    sim.method = method;
    place(sim, Vec3(1.2, 0.0, 0.0));
    sim.initialize();
    sim.run(T, dt);
    return sim.q(0);
}

} // namespace

TEST_CASE("Kernel forces: generalized forces are J^T times the body forces", "[kernel][forces]")
{
    Rng rng(61);
    const Model model = mbd_test::random_tree(rng, KJoint::Free, mbd_test::all_kernel_joints());
    Data data(model);
    const VecX q = mbd_test::random_configuration(model, rng);
    const VecX v = mbd_test::random_vector(model.nv, rng, 1.0);
    forward_kinematics(model, data, q, v);

    std::vector<RigidBodyForces> forces(static_cast<std::size_t>(model.nbodies()));
    for (std::size_t i = 1; i < forces.size(); ++i) {
        forces[i].f_W = rng.vec(10.0);
        forces[i].tau_W = rng.vec(5.0);
    }
    VecX tau;
    generalized_forces(model, data, forces, tau);

    VecX tau_ref = VecX::Zero(model.nv);
    MatX J;
    for (int i = 1; i < model.nbodies(); ++i) {
        body_jacobian_world(model, data, i, J);
        Vec6 f;
        f << forces[static_cast<std::size_t>(i)].tau_W, forces[static_cast<std::size_t>(i)].f_W;
        tau_ref += J.transpose() * f;
    }
    CHECK((tau - tau_ref).norm() < 1e-12 * (1.0 + tau_ref.norm()));

    // The body states are the world kinematics.
    for (int i = 1; i < model.nbodies(); ++i) {
        const RigidBodyState s = body_state(data, i);
        const Vec6 V = body_velocity_world(data, i);
        CHECK((s.p_WB - data.oMi[i].p).norm() == 0.0);
        CHECK((s.w_WB - V.head<3>()).norm() < 1e-15);
        CHECK((s.v_WB - V.tail<3>()).norm() < 1e-15);
    }
}

TEST_CASE("Kernel simulator: a mass on a spring oscillates at sqrt(k / m)", "[kernel][simulator]")
{
    // m = 2 kg, k = 50 N/m: omega = 5 rad/s; x(t) = 1 + 0.2 cos(5 t). RK4 at
    // h = 0.01 lags by N (omega h)^5 / 120 = 2.6e-7 rad after N = 100 steps,
    // so x is off by about 0.2 * 2.6e-7 = 5e-8.
    const Real T = 1.0;
    const Real exact = 1.0 + 0.2 * std::cos(5.0 * T);
    const Real e_rk4 = std::abs(spring_position(Integrator::RK4, 1e-2, T) - exact);
    CHECK(e_rk4 < 1e-7);

    // Convergence orders: halving dt divides the error by 2^4 (RK4) and by
    // 2 (semi-implicit Euler).
    const Real e_rk4_half = std::abs(spring_position(Integrator::RK4, 5e-3, T) - exact);
    CHECK(e_rk4 / e_rk4_half > 14.0);
    CHECK(e_rk4 / e_rk4_half < 18.0);

    const Real e_se = std::abs(spring_position(Integrator::SemiImplicitEuler, 1e-3, T) - exact);
    const Real e_se_half = std::abs(spring_position(Integrator::SemiImplicitEuler, 5e-4, T) - exact);
    CHECK(e_se / e_se_half > 1.8);
    CHECK(e_se / e_se_half < 2.2);
}

TEST_CASE("Kernel simulator: a constrained pendulum keeps its energy and its length",
          "[kernel][simulator]")
{
    System sys = free_body_system(1.0, Vec3(0.0, 0.0, -g_accel));
    sys.constraints.push_back(std::make_shared<Distance>(Marker{0, Transform3()}, Marker{1, Transform3()}, 1.0));
    kernel::Simulator sim(sys);
    place(sim, Vec3(std::sin(1.0), 0.0, -std::cos(1.0)));
    sim.initialize();

    auto energy = [&] {
        return kinetic_energy(sys.model, sim.data()) + potential_energy(sys.model, sim.data());
    };
    const Real E0 = energy();
    sim.run(2.0, 1e-3);
    // RK4 truncation and the projections, at 1 ms over 2 s: about 1e-9 J.
    CHECK(std::abs(energy() - E0) < 1e-8);
    CHECK(std::abs(sim.data().oMi[1].p.norm() - 1.0) < 1e-10);
    CHECK(sim.projection_failures() == 0);
    CHECK(sim.last_projection().converged);
    CHECK(std::abs(sim.time - 2.0) < 1e-12);
}

TEST_CASE("Kernel simulator: applied forces and callbacks", "[kernel][simulator]")
{
    // A 2 kg body, no gravity. A force of 4 N along X, applied through the
    // force callback, gives x = (F / m) t^2 / 2; RK4 is exact for it.
    System sys = free_body_system(2.0, Vec3::Zero());
    kernel::Simulator sim(sys);
    place(sim, Vec3::Zero());
    sim.initialize();

    int pre_force_calls = 0, post_step_calls = 0;
    sim.pre_force_callback = [&](kernel::Simulator&, Real) { ++pre_force_calls; };
    sim.post_step_callback = [&](kernel::Simulator&, Real) { ++post_step_calls; };
    sim.force_callback = [](kernel::Simulator&, Real, VecX& tau) { tau(3) += 4.0; };  // body X = world X

    sim.run(1.0, 0.01);
    CHECK(std::abs(sim.q(0) - 0.5 * 2.0 * 1.0 * 1.0) < 1e-12);
    CHECK(pre_force_calls == 4 * 100);   // four stages per RK4 step
    CHECK(post_step_calls == 100);

    // tau acts for one step only: an impulse of 6 N for 0.01 s along Y.
    sim.force_callback = nullptr;
    const Real vy0 = sim.v(4);
    sim.tau(4) = 6.0;
    sim.step(0.01);
    CHECK(std::abs(sim.v(4) - vy0 - 6.0 / 2.0 * 0.01) < 1e-14);
    CHECK(sim.tau.norm() == 0.0);
    sim.step(0.01);
    CHECK(std::abs(sim.v(4) - vy0 - 6.0 / 2.0 * 0.01) < 1e-14);
}

TEST_CASE("Kernel simulator: a driven pendulum follows a sine and reports the drive torque",
          "[kernel][simulator]")
{
    // A pendulum hinged about the world Y axis, its bob a distance L below the
    // hinge, driven along q(t) = A sin(w t). With I the inertia about the hinge,
    // the drive must supply tau = I q'' + m g L sin(q): the multiplier of the
    // driver (plan task 3.2).
    const Real m = 1.5, L = 0.8, A = 0.6, w = 3.0;
    System sys;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    RigidBodyInertia bob = RigidBodyInertia::from_solid_box(m, Vec3(0.05, 0.05, 0.05));
    bob.com_B = Vec3(0.0, 0.0, -L);
    const Transform3 z_to_y(Quat(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX())), Vec3::Zero());
    const int body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_to_y, z_to_y, bob);
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, body,
        TimeFunction([=](Real t) { return A * std::sin(w * t); },
                     [=](Real t) { return A * w * std::cos(w * t); },
                     [=](Real t) { return -A * w * w * std::sin(w * t); })));

    kernel::Simulator sim(sys);
    sim.initialize();   // makes v consistent with the drive: q_dot(0) = A w
    CHECK(std::abs(sim.v(0) - A * w) < 1e-12);

    const Real I_hinge = bob.I_com_B(1, 1) + m * L * L;
    Real worst_q = 0.0, worst_tau = 0.0;
    for (int step = 0; step < 2000; ++step) {
        sim.step(1e-3);
        const Real t = sim.time;
        worst_q = std::max(worst_q, std::abs(sim.q(0) - A * std::sin(w * t)));
        sim.acceleration(sim.q, sim.v, t);
        const Real tau_drive = sim.solver().lambda()(0);
        const Real q = sim.q(0), q_ddot = -A * w * w * std::sin(w * t);
        worst_tau = std::max(worst_tau, std::abs(tau_drive - (I_hinge * q_ddot + m * g_accel * L * std::sin(q))));
    }
    CHECK(worst_q < 1e-10);
    CHECK(worst_tau < 1e-8);
}

namespace {
int g_warnings = 0;
void count_warning(const std::string&) { ++g_warnings; }
}

TEST_CASE("Kernel simulator: a projection that cannot converge is counted and reported once",
          "[kernel][simulator]")
{
    // The same body point held at height 0 and at height 0.5: no
    // configuration satisfies both. The two equations have the same
    // Jacobian, so the solver keeps the first and drops the second as
    // redundant: the point stays at height 0, and the projection cannot
    // bring the second residual (0.5) down.
    System sys;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)));
    for (const Real height : {0.0, 0.5}) {
        sys.constraints.push_back(std::make_shared<Dot2>(
            Marker{0, Transform3()}, 2, Marker{1, Transform3()}, TimeFunction::constant(height)));
    }

    const DiagnosticSink previous_sink = diagnostic_sink();
    diagnostic_sink() = &count_warning;
    g_warnings = 0;

    kernel::Simulator sim(sys);
    sim.initialize();
    sim.run(0.05, 1e-3);
    diagnostic_sink() = previous_sink;

    CHECK_FALSE(sim.last_projection().converged);
    CHECK(sim.projection_failures() == 50);   // every step
    // One report for the redundancy, one for the first failed projection.
    CHECK(g_warnings == 2);

    // The first equation holds; the state stays finite.
    CHECK(std::abs(sim.states()[1].p_WB.z()) < 1e-9);
    CHECK(std::abs(sim.last_projection().position_residual - 0.5) < 1e-9);
    CHECK(sim.q.allFinite());
    CHECK(sim.v.allFinite());
}

TEST_CASE("Kernel simulator: a point driven along a circle follows it and reports the force",
          "[kernel][simulator]")
{
    // Plan task 3.2, a driven point. A free body's centre of mass is held on a
    // horizontal circle of radius R at height h, traversed at rate w, by three
    // dot-2 drivers on its coordinates. The point follows the circle exactly,
    // the body does not turn (the constraint force acts through the centre of
    // mass), and the constraint force is what the motion needs:
    // F = m a - m g = (-m w^2 R cos wt, -m w^2 R sin wt, m g).
    const Real m = 2.0, R = 0.5, w = 3.0, h = 1.0;
    System sys;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.2, 0.3)));
    const TimeFunction x([=](Real t) { return R * std::cos(w * t); },
                         [=](Real t) { return -R * w * std::sin(w * t); },
                         [=](Real t) { return -R * w * w * std::cos(w * t); });
    const TimeFunction y([=](Real t) { return R * std::sin(w * t); },
                         [=](Real t) { return R * w * std::cos(w * t); },
                         [=](Real t) { return -R * w * w * std::sin(w * t); });
    const TimeFunction z = TimeFunction::constant(h);
    int axis = 0;
    for (const TimeFunction& s : {x, y, z}) {
        sys.constraints.push_back(std::make_shared<Dot2>(Marker{0, Transform3()}, axis++,
                                                         Marker{1, Transform3()}, s));
    }

    Simulator sim(sys);
    sim.q.head<3>() = Vec3(R, 0.0, h);
    sim.v.tail<3>() = Vec3(0.0, R * w, 0.0);   // body axes are world axes at the start
    sim.initialize();

    Real worst_point = 0.0, worst_force = 0.0, worst_spin = 0.0;
    for (int step = 0; step < 1000; ++step) {
        sim.step(1e-3);
        const Real t = sim.time;
        const Vec3 target(R * std::cos(w * t), R * std::sin(w * t), h);
        worst_point = std::max(worst_point, (sim.states()[1].p_WB - target).norm());
        worst_spin = std::max(worst_spin, sim.v.head<3>().norm());
        if (step % 100 == 99) {
            sim.acceleration(sim.q, sim.v, t);
            const VecX f = sim.solver().J().transpose() * sim.solver().lambda();
            const Vec3 expected(-m * w * w * R * std::cos(w * t), -m * w * w * R * std::sin(w * t),
                                m * g_accel);
            // Generalized force of the free joint: the force at the body
            // origin in body axes, here the world's.
            worst_force = std::max(worst_force, (Vec3(f.tail<3>()) - expected).norm());
        }
    }
    CHECK(worst_point <= 1e-10);   // the projection holds |phi| <= 1e-10, and phi is the error
    CHECK(worst_spin < 1e-10);
    CHECK(worst_force < 1e-8 * m * g_accel);
}
