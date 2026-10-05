// Linearisation about an operating point (plan task 3.7).
//
//   - a damped oscillator: its natural frequency, damping ratio, reduced
//     mass, stiffness, damping and input matrix in closed form;
//   - the quarter car (the plan's "done when"): undamped frequencies against
//     the closed form, damped eigenvalues against the state matrix written
//     by hand, and the body-bounce mode's shape;
//   - a compound pendulum held by a revolute closure on a free body: its
//     restoring stiffness comes entirely from the constraint forces, and its
//     frequency is sqrt(m g d / I);
//   - a four-bar under gravity and a spring: the frequency of small
//     oscillations along the loop by the energy method, V'' / M_eff;
//   - an upright balance: the growing mode and its negative omega^2;
//   - an operating point that is not an equilibrium is reported, and the
//     simulator's state is left as it was;
//   - the detailed sedan: ten degrees of freedom, three without stiffness,
//     nothing unstable.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::FourBar;

namespace {

bool has_code(const std::vector<std::string>& messages, const std::string& code)
{
    for (const auto& m : messages) {
        if (m.rfind(code, 0) == 0) return true;
    }
    return false;
}

/// Joint frames whose Z axis (the slider's axis) is world Y, the body frame
/// kept aligned with the world.
Transform3 z_to_y()
{
    return Transform3::FromRotation(Mat3(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX()).toRotationMatrix()));
}

std::shared_ptr<JointCoordinateForce> torsion_spring(const Model& model, int body, Real k, Real reference)
{
    JointCoordinateForceParams p;
    p.spring = Curve::linear(k);
    p.reference = reference;
    return std::make_shared<JointCoordinateForce>(model, body, p);
}

Real relative(Real a, Real b) { return std::abs(a - b) / std::abs(b); }

} // namespace

TEST_CASE("Linearisation: a damped oscillator in closed form", "[kernel][linearization]")
{
    // A mass on a vertical slider over a spring-damper: m x'' + c x' + k x = u.
    // omega_n = sqrt(k / m), zeta = c / (2 sqrt(k m)). The forces are linear,
    // so the central differences are exact up to roundoff: eps times the
    // largest force over the step, 2e-16 x 50 N / 1e-5, a relative 1e-12 of
    // k; the bounds of 1e-9 leave a wide margin.
    const Real m = 2.0, k = 800.0, c = 12.0, L0 = 0.5;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const int b = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), z_to_y(), z_to_y(),
                                     RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.1)));
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0, b, Vec3::Zero(), Vec3::Zero(), k, c, L0));
    Simulator sim(sys);
    sim.q(0) = L0;
    REQUIRE(static_equilibrium(sim).converged);

    const Linearization L = linearize(sim);
    INFO(L.summary(sys.model));
    REQUIRE(L.degrees_of_freedom() == 1);
    REQUIRE(L.modes.size() == 1);
    const Real wn = std::sqrt(k / m), zeta = c / (2.0 * std::sqrt(k * m));
    CHECK(relative(L.modes[0].natural_frequency_hz, wn / (2.0 * pi)) <= 1e-9);
    CHECK(relative(L.modes[0].damping_ratio, zeta) <= 1e-9);
    CHECK(relative(L.modes[0].damped_frequency_hz, wn * std::sqrt(1.0 - zeta * zeta) / (2.0 * pi)) <= 1e-9);
    // N is +-1: the reduced matrices are m, k and c themselves.
    CHECK(relative(L.M(0, 0), m) <= 1e-12);
    CHECK(relative(L.K(0, 0), k) <= 1e-9);
    CHECK(relative(L.C(0, 0), c) <= 1e-9);
    // A generalized force u on the slider accelerates it by u / m, along N.
    CHECK(relative(L.B(1, 0) * L.N(0, 0), 1.0 / m) <= 1e-12);
    CHECK(L.B(0, 0) == 0.0);
    CHECK(L.warnings.empty());
}

TEST_CASE("Linearisation: quarter-car frequencies match the closed form", "[kernel][linearization]")
{
    // The quarter car of tests/test_quarter_car.cpp: wheel (40 kg) and body
    // (250 kg) on vertical sliders, suspension spring and damper between
    // them, tyre spring under the wheel. With x = (wheel, body):
    //   M = diag(mu, ms), K = [[ks + kt, -ks], [-ks, ks]], C = [[cs, -cs], [-cs, cs]].
    const Real ms = 250.0, mu = 40.0, ks = 20000.0, kt = 200000.0, cs = 1500.0, R = 0.35, L0 = 0.30;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const auto slider = std::make_shared<PrismaticJointModel>();
    sys.model.add_body(0, slider, z_to_y(), z_to_y(), RigidBodyInertia::from_solid_box(mu, Vec3(0.15, 0.15, 0.15)), "wheel");
    sys.model.add_body(0, slider, z_to_y(), z_to_y(), RigidBodyInertia::from_solid_box(ms, Vec3(0.5, 0.2, 0.4)), "body");
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(1, 2, Vec3::Zero(), Vec3::Zero(), ks, cs, L0));
    sys.force_elements.push_back(std::make_shared<TireContactForce>(1, R, kt, 0.0));
    Simulator sim(sys);
    sim.q << R - 0.01, R + 0.25;
    REQUIRE(static_equilibrium(sim).converged);

    const Linearization L = linearize(sim);
    INFO(L.summary(sys.model));
    REQUIRE(L.undamped.size() == 2);
    REQUIRE(L.modes.size() == 2);

    // Undamped: omega^4 - (a + b) omega^2 + (a b - ks^2 / (mu ms)) = 0 with
    // a = (ks + kt) / mu, b = ks / ms. Roundoff as for the oscillator, with
    // the tyre's force (3 kN) the largest: a relative 1e-11.
    const Real a = (ks + kt) / mu, b = ks / ms;
    const Real sum = a + b, product = a * b - (ks / mu) * (ks / ms);
    const Real disc = std::sqrt(sum * sum - 4.0 * product);
    const Real w1 = std::sqrt(0.5 * (sum - disc)), w2 = std::sqrt(0.5 * (sum + disc));
    CHECK(relative(L.undamped[0].frequency_hz, w1 / (2.0 * pi)) <= 1e-9);
    CHECK(relative(L.undamped[1].frequency_hz, w2 / (2.0 * pi)) <= 1e-9);

    // Body bounce: the body's row of (K - w1^2 M) x = 0 gives the wheel's
    // amplitude relative to the body's, (ks - w1^2 ms) / ks.
    const VecX& bounce = L.undamped[0].shape;
    CHECK(L.undamped[0].largest_at == 1);
    CHECK(std::abs(bounce(0) / bounce(1) - (ks - w1 * w1 * ms) / ks) <= 1e-9);

    // Damped: the eigenvalues of [[0, I], [-M^-1 K, -M^-1 C]], written here.
    Eigen::Matrix2d M, K, C;
    M << mu, 0.0, 0.0, ms;
    K << ks + kt, -ks, -ks, ks;
    C << cs, -cs, -cs, cs;
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    A.topRightCorner<2, 2>().setIdentity();
    A.bottomLeftCorner<2, 2>() = -M.inverse() * K;
    A.bottomRightCorner<2, 2>() = -M.inverse() * C;
    const Eigen::EigenSolver<Eigen::Matrix4d> es(A);
    for (const Mode& mode : L.modes) {
        Real nearest = 1e300;
        for (int i = 0; i < 4; ++i) nearest = std::min(nearest, std::abs(es.eigenvalues()(i) - mode.eigenvalue));
        CHECK(nearest <= 1e-9 * std::abs(mode.eigenvalue));
    }
    CHECK(L.warnings.empty());
    CHECK(L.notes.empty());
}

TEST_CASE("Linearisation: a pendulum held by a revolute closure", "[kernel][linearization]")
{
    // A free body hanging from a revolute closure at the origin, its centre
    // of mass d below the hinge. Its tree joint has no stiffness at all: the
    // restoring stiffness comes from the constraint forces turning with the
    // body, the term of J^T lambda that linearisation must not lose.
    // omega^2 = m g d / (I_zz + m d^2).
    const Real m = 3.0, d = 0.4;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, Vec3(0.05, 0.3, 0.08));
    const int body = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), I);
    sys.constraints.push_back(revolute_closure(Marker{0, Transform3()},
                                               Marker{body, Transform3::FromTranslation(Vec3(0.0, d, 0.0))}));
    Simulator sim(sys);
    sim.q.head<3>() = Vec3(0.0, -d, 0.0);   // hanging: the hinge point at the origin
    sim.initialize();

    const Linearization L = linearize(sim);
    INFO(L.summary(sys.model));
    REQUIRE(L.degrees_of_freedom() == 1);
    const Real w2 = m * g_accel * d / (I.I_com_B(2, 2) + m * d * d);
    // The forces are nonlinear in the angle (sin); the central difference's
    // truncation, h^2 / 6 of the relative third derivative, is 2e-11.
    CHECK(relative(L.undamped[0].omega_squared, w2) <= 1e-8);
    CHECK(L.operating_acceleration <= 1e-12);
}

TEST_CASE("Linearisation: a four-bar oscillates at V'' / M_eff along its loop", "[kernel][linearization]")
{
    // Small oscillations of a one-degree-of-freedom loop about its
    // equilibrium, by the energy method: with q = closed(th) the loop's
    // configurations, V(th) the potential energy and M_eff(th) = q'^T M q',
    // omega^2 = V''(th*) / M_eff(th*). V'' by a second difference of step
    // 1e-4 (truncation 1e-9 relative, roundoff eps V / H^2, 4e-8 relative),
    // q' by a central difference of step 1e-5 (1e-10): bound 1e-6.
    FourBar fb;
    System sys;
    sys.model = fb.model;
    sys.constraints = fb.closure;
    const Real k = 50.0, th0 = 1.0;
    sys.joint_forces.push_back(torsion_spring(sys.model, fb.crank, k, th0));
    Simulator sim(sys);
    sim.q = FourBar::closed(1.2);
    REQUIRE(static_equilibrium(sim).converged);
    const Real th = sim.q(0);

    const Linearization L = linearize(sim);
    INFO(L.summary(sys.model));
    REQUIRE(L.degrees_of_freedom() == 1);

    Data data(sys.model);
    auto V = [&](Real x) {
        forward_kinematics(sys.model, data, FourBar::closed(x), VecX::Zero(3));
        return potential_energy(sys.model, data) + 0.5 * k * (x - th0) * (x - th0);
    };
    const Real H = 1e-4, hq = 1e-5;
    const Real V2 = (V(th + H) - 2.0 * V(th) + V(th - H)) / (H * H);
    const VecX dq = (FourBar::closed(th + hq) - FourBar::closed(th - hq)) / (2.0 * hq);
    crba(sys.model, data, sim.q);
    const Real M_eff = dq.dot(data.M * dq);
    CHECK(relative(L.undamped[0].omega_squared, V2 / M_eff) <= 1e-6);
    CHECK(L.modes.size() == 1);
    CHECK(L.modes[0].damping_ratio == 0.0);   // nothing dissipates
}

TEST_CASE("Linearisation: an upright balance has a growing mode", "[kernel][linearization]")
{
    // A bob above its hinge on a spring too weak to hold it: stiffness
    // k - m g l < 0, omega^2 = (k - m g l) / I, and A has one eigenvalue
    // +sqrt(-omega^2).
    const Real m = 1.0, l = 0.5, k = 1.0;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, Vec3(0.02, 0.02, 0.02));
    I.com_B = Vec3(l, 0.0, 0.0);
    const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), I);
    sys.joint_forces.push_back(torsion_spring(sys.model, b, k, 0.5 * pi));
    Simulator sim(sys);
    sim.q(0) = 0.5 * pi;

    const Linearization L = linearize(sim);
    INFO(L.summary(sys.model));
    const Real I_hinge = I.I_com_B(2, 2) + m * l * l;
    const Real w2 = (k - m * g_accel * l) / I_hinge;
    CHECK(relative(L.undamped[0].omega_squared, w2) <= 1e-8);
    CHECK(has_code(L.warnings, "MBD-K082"));
    Real largest_real = -1e300;
    for (const Mode& mode : L.modes) largest_real = std::max(largest_real, mode.eigenvalue.real());
    CHECK(relative(largest_real, std::sqrt(-w2)) <= 1e-8);
}

TEST_CASE("Linearisation: an operating point off equilibrium is reported and left alone",
          "[kernel][linearization]")
{
    FourBar fb;
    System sys;
    sys.model = fb.model;
    sys.constraints = fb.closure;
    Simulator sim(sys);
    sim.q = fb.q;
    sim.v = VecX::Zero(3);
    sim.initialize();
    const VecX q = sim.q, v = sim.v, tau = sim.tau;

    const Linearization L = linearize(sim);
    CHECK(has_code(L.warnings, "MBD-K080"));   // gravity accelerates the loop
    CHECK(sim.q == q);
    CHECK(sim.v == v);
    CHECK(sim.tau == tau);
}

TEST_CASE("Linearisation: the sedan at rest has three free directions and nothing unstable",
          "[kernel][linearization][vehicle]")
{
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    System sys;
    const VehicleHandle vh = build_vehicle(sys, tmpl);
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    REQUIRE(static_equilibrium(sim).converged);

    const Linearization L = linearize(sim);
    INFO(L.summary(sys.model));
    CHECK(L.degrees_of_freedom() == 10);
    int zero = 0, negative = 0;
    for (const UndampedMode& u : L.undamped) {
        if (u.omega_squared == 0.0) ++zero;
        if (u.omega_squared < 0.0) ++negative;
    }
    CHECK(zero == 3);
    CHECK(negative == 0);
    CHECK(has_code(L.notes, "MBD-K081"));
    CHECK_FALSE(has_code(L.warnings, "MBD-K082"));
    CHECK_FALSE(has_code(L.warnings, "MBD-K080"));
}
