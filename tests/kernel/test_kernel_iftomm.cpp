// The IFToMM benchmark problems of group A (plan task 5.2, D44): the simple
// pendulum, the N-four-bar mechanism, Andrews' squeezing mechanism, Bricard's
// mechanism and the flyball governor (kernel/iftomm.hpp), each against the
// strongest reference available here:
//
//   - the pendulum and the N-four-bar against their closed forms (Jacobi's
//     functions: the four-bar moves as one parallelogram, a pendulum turning
//     over and over, through a singular position every half turn);
//   - Andrews' mechanism against the reference solution the Test Set for IVP
//     Solvers publishes at t = 0.03 s;
//   - Bricard's mechanism against the closed form of its motion (the
//     relation between its two joint angles, derived from the closure
//     equations) and against its dynamics reduced to one coordinate, written
//     here from the geometry alone and integrated on its own;
//   - the flyball governor, whose stiff springs and strong dampers are the
//     point, against the angular momentum about its spindle (exactly kept),
//     its energy balance (the energy lost equals the dampers' work) and the
//     agreement of three integrators.
//
// The IFToMM library's own reference histories (www.iftomm-multibody.org)
// could not be fetched from the machine this was written on; the closed
// forms replace them for A01, A02 and A04, the test set's solution is A03's,
// and A04's and A05's comparison with them is open (journal: Phase 5, IFToMM
// benchmarks).
//
// The N-four-bar needs Simulator::constraint_tolerance at 1e-14 (D44): with
// the default 1e-10 the positions keep a residual of about 1e-13 that the
// projection leaves, the linkage behaves as a slightly imperfect
// parallelogram, and near a singular position its coupler must turn at a rate
// that grows as delta theta' / (L theta^2); the step size control then cannot
// pass. The other problems are run the same way, for comparable results.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"
#include "mbd/kernel/validate.hpp"

#include "kernel/iftomm.hpp"

using namespace mbd;
using namespace mbd::kernel;
using namespace mbd_test::iftomm;

namespace {

/// A simulator on `sys` with Dormand-Prince or Radau IIA at tolerance `tol`
/// and the constraints held to 1e-14.
void configure(Simulator& sim, Integrator method, Real tol)
{
    sim.method = method;
    sim.step_control.relative_tolerance = tol;
    sim.step_control.absolute_tolerance = tol;
    sim.constraint_tolerance = 1e-14;
}

Real total_energy(const Model& model, Data& data, const VecX& q, const VecX& v)
{
    forward_kinematics(model, data, q, v);
    return kinetic_energy(model, data) + potential_energy(model, data);
}

// --- Bricard's mechanism reduced to one coordinate ------------------------------------

/// Bricard's mechanism as a system of one coordinate phi, from its geometry
/// only: the joint angles from the closed form (Bricard::psi), each bar's
/// placement by the product of the turns about the joints' axes as they are at
/// the start (each a fixed world axis through a cube corner, moved by the
/// turns before it), its velocity from the joints' rates. Then the kinetic
/// energy J(phi) phi'^2 / 2 and the potential energy V(phi) give
///
///   phi'' = -(J'(phi) phi'^2 / 2 + V'(phi)) / J(phi),
///
/// J' and V' by central differences at 1e-5 rad (error h^2 / 6 times the
/// third derivative, about 1e-11 of J here).
class BricardReduced {
public:
    BricardReduced()
    {
        const Vec3 corner[6] = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, -1, 0), Vec3(1, -1, 1), Vec3(0, -1, 1),
                                Vec3(0, 0, 1)};
        const int axis[5] = {1, 2, 0, 1, 2};
        for (int i = 0; i < 5; ++i) {
            point_[i] = corner[i];
            axis_[i] = Vec3::Unit(axis[i]);
            centre_[i] = 0.5 * (corner[i] + corner[i + 1]);
            const Vec3 direction = (corner[i + 1] - corner[i]).cwiseAbs();
            inertia_[i] = Bricard::box(centre_[i], direction).I_com_B;
        }
    }

    /// J(phi) and V(phi).
    void energies(Real phi, Real& J, Real& V) const
    {
        const Real u = std::tan(0.5 * phi);
        const Real psi = Bricard::psi(phi);
        const Real w = std::tan(0.5 * psi);
        // dpsi/dphi by implicit differentiation of u w^2 + (u + 1)^2 w + u = 0.
        const Real dw_du = -(w * w + 2.0 * (u + 1.0) * w + 1.0) / (2.0 * u * w + (u + 1.0) * (u + 1.0));
        const Real dpsi = (2.0 / (1.0 + w * w)) * dw_du * (0.5 * (1.0 + u * u));
        const Real angle[5] = {phi, psi, phi, psi, phi};
        const Real rate[5] = {1.0, dpsi, 1.0, dpsi, 1.0};   // per unit phi'
        Mat3 R = Mat3::Identity();
        Vec3 p = Vec3::Zero();                              // the placement x -> R x + p so far
        Vec3 axis_now[5], point_now[5];
        J = 0.0;
        V = 0.0;
        for (int k = 0; k < 5; ++k) {
            axis_now[k] = R * axis_[k];
            point_now[k] = R * point_[k] + p;
            const Mat3 turn = Eigen::AngleAxisd(angle[k], axis_[k]).toRotationMatrix();
            p = R * (point_[k] - turn * point_[k]) + p;    // x -> R (turn (x - c) + c) + p
            R = R * turn;
            const Vec3 c = R * centre_[k] + p;
            Vec3 omega = Vec3::Zero(), velocity = Vec3::Zero();
            for (int i = 0; i <= k; ++i) {
                omega += axis_now[i] * rate[i];
                velocity += axis_now[i].cross(c - point_now[i]) * rate[i];
            }
            J += Bricard::m * velocity.squaredNorm() + omega.dot(R * inertia_[k] * R.transpose() * omega);
            V += Bricard::m * Bricard::g * c.y();
        }
    }

    Real acceleration(Real phi, Real phi_dot) const
    {
        const Real h = 1e-5;
        Real J, V, Jp, Vp, Jm, Vm;
        energies(phi, J, V);
        energies(phi + h, Jp, Vp);
        energies(phi - h, Jm, Vm);
        return -(0.5 * (Jp - Jm) / (2.0 * h) * phi_dot * phi_dot + (Vp - Vm) / (2.0 * h)) / J;
    }

private:
    Vec3 point_[5], axis_[5], centre_[5];
    Mat3 inertia_[5];
};

// --- The flyball governor's dissipated work ----------------------------------------------

/// One state: the work done by the governor's two dampers, integrated with
/// the bodies: rate c (dL/dt)^2 for each spring-damper of length L.
class DampersWork final : public StateElement {
public:
    explicit DampersWork(const Flyball& f) : f_(f) {}
    const char* name() const override { return "dampers' work"; }
    int size() const override { return 1; }
    void initial_states(VecRef x) const override { x(0) = 0.0; }
    void begin_evaluation(const Simulator&, Real, ConstVecRef) override {}
    void rates(const Simulator& sim, Real, ConstVecRef, VecRef x_dot) override
    {
        const auto& st = sim.states();
        Real power = 0.0;
        for (int side = 0; side < 2; ++side) {
            const Real s = side == 0 ? 1.0 : -1.0;
            const auto& a = st[static_cast<std::size_t>(f_.collar)];
            const auto& b = st[static_cast<std::size_t>(f_.rod[side])];
            const Vec3 ra = a.q_WB * Vec3(s * Flyball::hinge_offset, 0.0, 0.0);
            const Vec3 rb = b.q_WB * Vec3(0.0, 0.0, 0.5 * Flyball::rod_length);
            const Vec3 d = (b.p_WB + rb) - (a.p_WB + ra);
            const Vec3 d_dot = (b.v_WB + b.w_WB.cross(rb)) - (a.v_WB + a.w_WB.cross(ra));
            const Real L_dot = d.dot(d_dot) / d.norm();
            power += Flyball::c * L_dot * L_dot;
        }
        x_dot(0) = power;
    }

private:
    const Flyball& f_;
};

/// The governor's energy: kinetic, gravity's and the springs'.
Real flyball_energy(const Flyball& f, Data& data, const VecX& q, const VecX& v)
{
    const Model& model = f.sys.model;
    Real E = total_energy(model, data, q, v);
    for (int side = 0; side < 2; ++side) {
        const Real s = side == 0 ? 1.0 : -1.0;
        const Transform3& A = data.oMi[static_cast<std::size_t>(f.collar)];
        const Transform3& B = data.oMi[static_cast<std::size_t>(f.rod[side])];
        const Real L = ((B.p + B.q * Vec3(0.0, 0.0, 0.5 * Flyball::rod_length))
                        - (A.p + A.q * Vec3(s * Flyball::hinge_offset, 0.0, 0.0)))
                           .norm();
        E += 0.5 * Flyball::k * (L - Flyball::free_length) * (L - Flyball::free_length);
    }
    return E;
}

/// Angular momentum about the vertical axis.
Real momentum_z(const Model& model, Data& data, const VecX& q, const VecX& v)
{
    forward_kinematics(model, data, q, v);
    return momentum_world(model, data)(2);
}

} // namespace

// --- A01 -------------------------------------------------------------------------------

TEST_CASE("IFToMM A01: a simple pendulum follows the closed form for 10 s, on a joint or on a constraint",
          "[kernel][iftomm]")
{
    // The mass's position every 10 ms against Pendulum::exact, over the 5
    // periods of 10 s, swinging through 90 degrees, with Dormand-Prince at
    // 1e-10 and at 1e-12. Its error control is local, so the global error is
    // a model-dependent multiple of the tolerance (0.287 T tol on the
    // oscillator, task 4.2); measured here: 1.1e-9 and 8.6e-11 m on the joint,
    // 1.4e-8 and 1.9e-10 m on the constraint, where the free body's six
    // velocities and the projection after each step add to what the error
    // control sees. Allowed: 5e-8 m at 1e-10 and 1e-9 m at 1e-12, the energy
    // likewise in J, and the error must fall at least fivefold.
    for (bool as_constraint : {false, true}) {
        Real worst[2] = {0.0, 0.0}, worst_energy[2] = {0.0, 0.0};
        for (int run = 0; run < 2; ++run) {
            Pendulum p(as_constraint);
            Simulator sim(p.sys);
            configure(sim, Integrator::DormandPrince, run == 0 ? 1e-10 : 1e-12);
            p.release(sim);
            sim.initialize();
            Data data(p.sys.model);
            const Real E0 = total_energy(p.sys.model, data, sim.q, sim.v);
            for (int k = 1; k <= 1000; ++k) {
                sim.step(0.01);
                worst[run] = std::max(worst[run], (p.position(sim) - Pendulum::exact(sim.time)).norm());
                worst_energy[run] =
                    std::max(worst_energy[run], std::abs(total_energy(p.sys.model, data, sim.q, sim.v) - E0));
            }
        }
        INFO((as_constraint ? "on a distance constraint" : "on a revolute joint")
             << ": largest distance from the closed form " << worst[0] << " and " << worst[1]
             << " m at tolerances 1e-10 and 1e-12; energy change " << worst_energy[0] << " and " << worst_energy[1]
             << " J");
        CHECK(worst[0] < 5e-8);
        CHECK(worst[1] < 1e-9);
        CHECK(worst[1] < 0.2 * worst[0]);
        CHECK(worst_energy[0] < 5e-8);
        CHECK(worst_energy[1] < 1e-9);
    }
}

// --- A02 -------------------------------------------------------------------------------

namespace {

/// The largest |theta - theta_exact| over 10 s, sampled every 10 ms, the
/// constraints held to `constraint_tolerance`.
Real n_four_bar_error(int N, Real tol, Real constraint_tolerance = 1e-14)
{
    NFourBar fb(N);
    Simulator sim(fb.sys);
    configure(sim, Integrator::DormandPrince, tol);
    sim.constraint_tolerance = constraint_tolerance;
    fb.release(sim);
    sim.initialize();
    Real worst = 0.0;
    for (int k = 1; k <= 1000; ++k) {
        sim.step(0.01);
        worst = std::max(worst, std::abs(NFourBar::theta(sim) - fb.exact(sim.time)));
        // The couplers stay level: every coupler at minus the bar below it.
        REQUIRE(std::abs(sim.q(0) + sim.q(1)) < 1e-6);
    }
    return worst;
}

} // namespace

TEST_CASE("IFToMM A02: the N-four-bar mechanism turns through its singular positions as the closed form says",
          "[kernel][iftomm]")
{
    // Four four-bars in a row (the closed form holds for any N; N = 40 is the
    // hidden case below), started upright at -1 rad/s: 4.8 turns in 10 s,
    // through the singular horizontal position 19 times, each time with the
    // constraint equations falling from rank 8 to 4. Error against the
    // closed form, measured: 2.1e-8 rad at tolerance 1e-10 and 3.1e-9 at
    // 1e-12, growing in the slow passes over the top, where the linkage is
    // near its unstable upright equilibrium (errors grow there as e^(3.4 t)),
    // and with no step at the singular passes. Allowed 1e-7 and 1.5e-8 rad:
    // the error must also fall with the tolerance.
    const Real e10 = n_four_bar_error(4, 1e-10);
    const Real e12 = n_four_bar_error(4, 1e-12);
    INFO("largest error " << e10 << " rad at tolerance 1e-10, " << e12 << " at 1e-12");
    CHECK(e10 < 1e-7);
    CHECK(e12 < 1.5e-8);
    CHECK(e12 < 0.5 * e10);
}

TEST_CASE("IFToMM A02 with 40 four-bars", "[.][kernel][iftomm][long]")
{
    // The library's size: 81 bodies, 120 constraint equations, coordinates
    // up to 40 m. The projection's floor is then near 1e-12 m (residuals up
    // to 8.8e-13 when 1e-14 was asked, the projections failing with MBD-K051),
    // so the constraints are held to 1e-12; the error grows accordingly
    // (measured 3.4e-7 rad; 1.8e-7 with 1e-14 asked). Allowed 1e-6 rad. About
    // 17 s.
    const Real e = n_four_bar_error(40, 1e-10, 1e-12);
    INFO("largest error " << e << " rad");
    CHECK(e < 1e-6);
}

// --- A03 -------------------------------------------------------------------------------

TEST_CASE("IFToMM A03: Andrews' mechanism reaches the published reference at 0.03 s", "[kernel][iftomm]")
{
    // At the start the accelerations must be the test set's (its initial
    // values: beta'' = 14222.4439199541, Theta'' = -10666.8329399656 rad/s^2,
    // the others zero), to rounding grown by the solve (1e-9 relative).
    //
    // At 0.03 s: Radau IIA at 1e-10 and Dormand-Prince at 1e-12 agree within
    // 1.2e-12 rad, so this side has converged; both differ from the
    // reference by at most 1.43e-9 rad (beta, 15.8 rad, by 1.1e-9: 7e-11
    // relative), and the difference is one shift in time: every coordinate's
    // error divided by its velocity is -1.005e-12 s, the same to three
    // digits. The motion is the published one, reached 1e-12 s later, as an
    // integration's phase error would make it (the reference: PSIDE at
    // tolerances 1e-14). Allowed 5e-9 rad, with the agreement of the two
    // methods held to 1e-10.
    Andrews a;
    VecX qs[2];
    for (int run = 0; run < 2; ++run) {
        Simulator sim(a.sys);
        if (run == 0) configure(sim, Integrator::RadauIIA, 1e-10);
        else configure(sim, Integrator::DormandPrince, 1e-12);
        a.release(sim);
        sim.initialize();
        if (run == 0) {
            const VecX acc = sim.acceleration(sim.q, sim.v, 0.0);
            INFO("initial accelerations " << acc.transpose());
            CHECK(std::abs(acc(0) - 14222.4439199541138705911625887) < 1e-9 * 14222.44);
            CHECK(std::abs(acc(1) + 10666.8329399655854029433719415) < 1e-9 * 14222.44);
            CHECK(acc.tail(5).norm() < 1e-9 * 14222.44);
        }
        sim.run(0.03, 0.001);
        qs[run] = sim.q;
    }
    const VecX error = qs[0] - Andrews::reference();
    INFO("Radau IIA: " << qs[0].transpose() << "\nerror " << error.transpose() << "\nDormand-Prince - Radau IIA "
                       << (qs[1] - qs[0]).transpose());
    CHECK(error.cwiseAbs().maxCoeff() < 5e-9);
    CHECK((qs[1] - qs[0]).cwiseAbs().maxCoeff() < 1e-10);
}

// --- A04 -------------------------------------------------------------------------------

TEST_CASE("IFToMM A04: Bricard's mechanism moves with one degree of freedom, on its closed-form path",
          "[kernel][iftomm]")
{
    // Grubler's count gives none; one closure equation is redundant (rank 4
    // of 5) and the mechanism, released at rest in its cube configuration,
    // swings under gravity from phi = 0 to phi = pi and back, in about 5 s.
    //
    // Along the motion, every 10 ms for 10 s: joints 1, 3 and 5 at one angle
    // phi and joints 2 and 4 at Bricard::psi(phi), the closed form; the
    // simulation holds the closure to 1e-14 m, so these hold to that over the
    // lever arms of 1 m (1e-12 rad allowed; seen 9e-15). Energy kept within
    // 1e-9 J (Dormand-Prince at 1e-10, a conservative system; seen 1.5e-10).
    // And phi(t) against BricardReduced integrated by RK4 at 0.1 ms ((w h)^4
    // ~ 1e-15 a step for the swing's rates of a few rad/s): seen 2.0e-9 rad,
    // Dormand-Prince's error at 1e-10; allowed 1e-7.
    Bricard b;
    {
        const ValidationReport report = validate(b.sys, VecX::Zero(5));
        INFO(report.summary());
        CHECK(report.degrees_of_freedom == 1);
        CHECK(report.redundant_equations == 1);
    }
    Simulator sim(b.sys);
    configure(sim, Integrator::DormandPrince, 1e-10);
    sim.q.setZero();
    sim.v.setZero();
    sim.initialize();
    Data data(b.sys.model);
    const Real E0 = total_energy(b.sys.model, data, sim.q, sim.v);

    const BricardReduced reduced;
    Real phi = 0.0, phi_dot = 0.0;
    auto rk4 = [&](Real h) {
        const Real k1p = phi_dot, k1v = reduced.acceleration(phi, phi_dot);
        const Real k2p = phi_dot + 0.5 * h * k1v, k2v = reduced.acceleration(phi + 0.5 * h * k1p, k2p);
        const Real k3p = phi_dot + 0.5 * h * k2v, k3v = reduced.acceleration(phi + 0.5 * h * k2p, k3p);
        const Real k4p = phi_dot + h * k3v, k4v = reduced.acceleration(phi + h * k3p, k4p);
        phi += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        phi_dot += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    };

    Real worst_path = 0.0, worst_energy = 0.0, worst_reduced = 0.0, largest = 0.0;
    for (int k = 1; k <= 1000; ++k) {
        sim.step(0.01);
        for (int j = 0; j < 100; ++j) rk4(1e-4);
        const VecX& q = sim.q;
        const Real psi = Bricard::psi(q(0));
        worst_path = std::max({worst_path, std::abs(q(2) - q(0)), std::abs(q(4) - q(0)), std::abs(q(1) - psi),
                               std::abs(q(3) - psi)});
        worst_energy = std::max(worst_energy, std::abs(total_energy(b.sys.model, data, sim.q, sim.v) - E0));
        worst_reduced = std::max(worst_reduced, std::abs(q(0) - phi));
        largest = std::max(largest, std::abs(q(0)));
    }
    INFO("largest phi " << largest << " rad; off the closed-form path by " << worst_path << " rad; energy change "
                        << worst_energy << " J; from the reduced model " << worst_reduced << " rad");
    CHECK(largest > 2.5);   // a large motion, not a small swing
    CHECK(worst_path < 1e-12);
    CHECK(worst_energy < 1e-9);
    CHECK(worst_reduced < 1e-7);
}

// --- A05 -------------------------------------------------------------------------------

TEST_CASE("IFToMM A05: the flyball governor keeps its angular momentum and its energy balance",
          "[kernel][iftomm]")
{
    // 10 s with Radau IIA at 1e-8 (the springs' 1.6e3 rad/s against a slow
    // motion of a few rad/s: stiff). Exactly kept by the physics: the angular
    // momentum about Z (no torque about the spindle) and E(t) + W(t) = E(0),
    // W the dampers' work, integrated as a state with the bodies. Both to the
    // integrator's tolerance: seen 1.1e-10 and 1.0e-10 relative to L(0) and
    // E(0); allowed 1e-6 of L(0), and 1e-7 of E(0), 24 uJ, well below the
    // dampers' work (3.1 mJ), so that the balance fails if that work is left
    // out.
    // Generalised-alpha at 1 ms and Dormand-Prince at 1e-8 must reach the
    // same state after 10 s: within 1e-5 of Radau IIA's in every coordinate.
    Flyball f;
    DampersWork work(f);
    Simulator sim(f.sys);
    configure(sim, Integrator::RadauIIA, 1e-8);
    sim.attach(work);
    f.release(sim);
    sim.initialize();
    Data data(f.sys.model);
    const Real L0 = momentum_z(f.sys.model, data, sim.q, sim.v);
    const Real E0 = flyball_energy(f, data, sim.q, sim.v);
    Real worst_L = 0.0, worst_balance = 0.0;
    for (int k = 1; k <= 100; ++k) {
        sim.step(0.1);
        worst_L = std::max(worst_L, std::abs(momentum_z(f.sys.model, data, sim.q, sim.v) - L0));
        worst_balance = std::max(worst_balance, std::abs(flyball_energy(f, data, sim.q, sim.v) + sim.x(0) - E0));
        CHECK(std::abs(sim.q(2) - sim.q(3)) < 1e-9);   // the two rods mirror each other
    }
    INFO("L(0) = " << L0 << " N m s, largest change " << worst_L << "; E(0) = " << E0 << " J, work of the dampers "
                   << sim.x(0) << " J, largest imbalance " << worst_balance << " J");
    CHECK(worst_L < 1e-6 * std::abs(L0));
    CHECK(worst_balance < 1e-7 * std::abs(E0));
    CHECK(sim.x(0) > 10.0 * 1e-7 * std::abs(E0));   // the dampers' work is what the balance checks

    // The other methods after 10 s: Dormand-Prince at 1e-8 (explicit: it
    // must resolve the springs, 1.2e5 evaluations; seen 1.4e-9 from Radau
    // IIA, allowed 1e-6), and generalised-alpha, second order, at fixed 1 ms
    // and 0.5 ms, whose distance from Radau IIA's state must fall by 4 to
    // within the next term (seen 2.8e-5 and 7.1e-6, ratio 3.90).
    auto state_after = [&](Integrator method, Real h) {
        Simulator other(f.sys);
        configure(other, method, 1e-8);
        f.release(other);
        other.initialize();
        other.run(10.0, h);
        return VecX(other.q);
    };
    const Real dp = (state_after(Integrator::DormandPrince, 0.1) - sim.q).cwiseAbs().maxCoeff();
    const Real ga1 = (state_after(Integrator::GeneralizedAlpha, 1e-3) - sim.q).cwiseAbs().maxCoeff();
    const Real ga2 = (state_after(Integrator::GeneralizedAlpha, 5e-4) - sim.q).cwiseAbs().maxCoeff();
    INFO("from Radau IIA after 10 s: Dormand-Prince " << dp << "; generalised-alpha " << ga1 << " at 1 ms, " << ga2
                                                       << " at 0.5 ms (ratio " << ga1 / ga2 << ")");
    CHECK(dp < 1e-6);
    CHECK(ga1 / ga2 > 3.5);
    CHECK(ga1 / ga2 < 4.5);
}
