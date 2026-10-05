// Assembly of initial conditions (plan task 3.4).
//
//   - two sliders in series under one linear constraint: the least correction
//     in the kinetic-energy metric is known in closed form at every level;
//   - a four-bar given inconsistent positions, velocities and accelerations,
//     with its crank held, is assembled onto the analytic closure, its
//     motion's derivatives, and reported (the plan's "done when");
//   - with nothing held, the correction is the least one: it satisfies the
//     optimality condition M d = J^T mu, and a brute-force search along the
//     loop's one degree of freedom finds the same point;
//   - held values that contradict the constraints are reported, not hidden;
//   - a rotation held whole on a free body is kept exactly;
//   - malformed hold lists are refused with their codes;
//   - a simulator is assembled at its time, drivers included, and warns when
//     it cannot be.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using mbd_test::FourBar;

namespace {

System four_bar_system()
{
    FourBar fb;
    System sys;
    sys.model = fb.model;
    sys.constraints = fb.closure;
    return sys;
}

Real phi_norm(const System& sys, const VecX& q, Real t)
{
    Data data(sys.model);
    ConstraintSolver solver(sys.model, sys.constraints);
    solver.evaluate(data, q, VecX::Zero(sys.model.nv), t);
    return solver.phi().norm();
}

MatX jacobian(const System& sys, const VecX& q, Real t)
{
    Data data(sys.model);
    ConstraintSolver solver(sys.model, sys.constraints);
    solver.evaluate(data, q, VecX::Zero(sys.model.nv), t);
    return solver.J();
}

/// Smallest singular value of the columns of J that are not held: how much
/// a residual |phi| can move the free coordinates, |dq| <= |phi| / s_min.
Real smallest_singular_value(const MatX& J, const std::vector<int>& free_columns)
{
    MatX Jf(J.rows(), static_cast<Index>(free_columns.size()));
    for (std::size_t k = 0; k < free_columns.size(); ++k) Jf.col(static_cast<Index>(k)) = J.col(free_columns[k]);
    const Eigen::JacobiSVD<MatX> svd(Jf);
    const auto& s = svd.singularValues();
    Real smallest = s(0);
    for (Index k = 0; k < s.size(); ++k) {
        if (s(k) > 1e-9 * s(0)) smallest = s(k);   // the nonzero ones
    }
    return smallest;
}

bool has_code(const std::vector<std::string>& messages, const std::string& code)
{
    for (const auto& m : messages) {
        if (m.rfind(code, 0) == 0) return true;
    }
    return false;
}

std::vector<std::string>& captured_warnings()
{
    static std::vector<std::string> w;
    return w;
}

} // namespace

TEST_CASE("Assembly: two sliders in series move by the least correction", "[kernel][assembly]")
{
    // Slider 1 on the ground and slider 2 on slider 1, both along Z, so body
    // 2 sits at z2 = x1 + x2. One driven equation holds it at s(t) = L + c t:
    // phi = x1 + x2 - s, J = [1 1], nu = c, gamma = 0. The mass matrix is
    // M = [[m1 + m2, m2], [m2, m2]]: moving x1 carries both bodies. The least
    // correction d of the residual r, M d = J^T mu with J d = r, is
    //     d = M^-1 J^T (J M^-1 J^T)^-1 r = (0, r):
    // M^-1 J^T = (0, 1/m2), J M^-1 J^T = 1/m2. Only slider 2 moves, since
    // moving it alone costs the least kinetic energy. Held at x2, x1 moves by r.
    const Real m1 = 3.0, m2 = 1.0, L = 0.5, c = 0.3, t = 1.0;
    System sys;
    const auto slide = std::make_shared<PrismaticJointModel>();
    const Transform3 I3;
    const int b1 = sys.model.add_body(0, slide, I3, I3, RigidBodyInertia::from_solid_box(m1, Vec3(0.1, 0.1, 0.1)), "slider 1");
    const int b2 = sys.model.add_body(b1, slide, I3, I3, RigidBodyInertia::from_solid_box(m2, Vec3(0.1, 0.1, 0.1)), "slider 2");
    sys.constraints.push_back(std::make_shared<Dot2>(
        Marker{0, I3}, 2, Marker{b2, I3},
        TimeFunction([=](Real tt) { return L + c * tt; }, [=](Real) { return c; }, [](Real) { return 0.0; })));

    const Real s = L + c * t;
    const Real exact = 1e-14;   // the constraint is linear: one step, then roundoff

    SECTION("nothing held")
    {
        VecX q(2), v(2), a(2);
        q << 0.1, 0.2;
        v << 1.0, 0.0;
        a << 2.0, 0.0;
        const Real r = s - (q(0) + q(1));
        const AssemblyReport rep = assemble(sys, q, v, a, t);
        CHECK(rep.ok());
        CHECK(std::abs(q(0) - 0.1) <= exact);
        CHECK(std::abs(q(1) - (0.2 + r)) <= exact);
        // J v = c: the velocity of body 2 becomes c, all of it in v2.
        CHECK(std::abs(v(0) - 1.0) <= exact);
        CHECK(std::abs(v(1) - (c - 1.0)) <= exact);
        // J a = 0, likewise.
        CHECK(std::abs(a(0) - 2.0) <= exact);
        CHECK(std::abs(a(1) + 2.0) <= exact);

        CHECK(rep.degrees_of_freedom == 1);
        CHECK(rep.position.freedom_left == 1);
        CHECK(has_code(rep.notes, "MBD-K066"));
        CHECK(rep.position.largest_change_at == 1);
        CHECK(std::abs(rep.position.largest_change - r) <= exact);
        CHECK(std::abs(rep.position.residual_before - r) <= exact);
    }

    SECTION("slider 2 held")
    {
        VecX q(2), v(2);
        q << 0.1, 0.2;
        v << 1.0, 0.0;
        AssemblySpec spec;
        spec.hold(sys.model, b2);
        const AssemblyReport rep = assemble(sys, q, v, t, spec);
        CHECK(rep.ok());
        CHECK(q(1) == 0.2);   // held: not touched at all
        CHECK(std::abs(q(0) - (s - 0.2)) <= exact);
        CHECK(v(1) == 0.0);
        CHECK(std::abs(v(0) - c) <= exact);
        CHECK(rep.position.freedom_left == 0);
        CHECK(rep.notes.empty());
        CHECK(rep.warnings.empty());
    }
}

TEST_CASE("Assembly: a four-bar given inconsistent values is assembled and reported",
          "[kernel][assembly]")
{
    // The crank is held at 1 rad, turning at 2 rad/s without acceleration; the
    // coupler and rocker angles are 0.25 and 0.3 rad off and their rates and
    // accelerations zero. The loop has one degree of freedom, which the crank
    // fixes: the assembled state is the analytic closure q(th) at th = 1, its
    // velocities q'(th) th_dot and its accelerations q''(th) th_dot^2.
    const System sys = four_bar_system();
    const Real th = 1.0, th_dot = 2.0;
    VecX q = FourBar::closed(th) + Eigen::Vector3d(0.0, 0.25, -0.3);
    VecX v = Eigen::Vector3d(th_dot, 0.0, 0.0);
    VecX a = VecX::Zero(3);
    const Real phi_before = phi_norm(sys, q, 0.0);

    AssemblySpec spec;
    spec.hold(sys.model, 1);   // the crank
    const AssemblyReport rep = assemble(sys, q, v, a, 0.0, spec);
    INFO(rep.summary(sys.model));

    REQUIRE(rep.ok());
    // Held coordinates are not touched at all.
    CHECK(q(0) == th);
    CHECK(v(0) == th_dot);
    CHECK(a(0) == 0.0);

    // Positions. The solver stops at |phi| <= tolerance; the free coordinates
    // can then be off by at most |phi| / s_min of their Jacobian columns.
    CHECK(rep.position.residual <= spec.tolerance);
    const Real s_min = smallest_singular_value(jacobian(sys, q, 0.0), {1, 2});
    const Real q_bound = 2.0 * spec.tolerance / s_min;
    CHECK((q - FourBar::closed(th)).cwiseAbs().maxCoeff() <= q_bound);

    // Velocities against central differences of the closure, step h = 1e-5:
    // truncation h^2 |q'''| / 6, about 1e-11 with |q'''| = 0.5 here, and
    // roundoff eps |q| / h, about 3e-11; times th_dot. Measured: 1.6e-11.
    const Real h = 1e-5;
    const VecX dq = (FourBar::closed(th + h) - FourBar::closed(th - h)) / (2.0 * h);
    CHECK((v - th_dot * dq).cwiseAbs().maxCoeff() <= 1e-8);

    // Accelerations against second differences, step H = 1e-3: truncation
    // H^2 |q''''| / 12, about 9e-8 with |q''''| = 1.1 here, and roundoff
    // 4 eps |q| / H^2, about 2e-9; times th_dot^2 = 4. Measured: 3.7e-7, the
    // truncation.
    const Real H = 1e-3;
    const VecX ddq = (FourBar::closed(th + H) - 2.0 * FourBar::closed(th) + FourBar::closed(th - H)) / (H * H);
    CHECK((a - th_dot * th_dot * ddq).cwiseAbs().maxCoeff() <= 4e-6);

    // The report.
    CHECK(rep.constraint_equations == 5);
    CHECK(rep.independent_equations == 2);
    CHECK(rep.degrees_of_freedom == 1);
    CHECK(rep.position.held == 1);
    CHECK(rep.position.freedom_left == 0);
    CHECK(std::abs(rep.position.residual_before - phi_before) <= 1e-15 * phi_before);
    CHECK(rep.position.residual_before > 0.1);
    CHECK((rep.position.largest_change_at == 1 || rep.position.largest_change_at == 2));
    CHECK(rep.velocity.residual <= spec.tolerance);
    CHECK(rep.acceleration.residual <= spec.tolerance);
    CHECK(rep.warnings.empty());
    CHECK(rep.notes.empty());
    const std::string text = rep.summary(sys.model);
    CHECK_THAT(text, ContainsSubstring("degrees of freedom 1"));
    CHECK_THAT(text, ContainsSubstring("positions: 1 held, 0 degree(s) of freedom left"));
    CHECK_THAT(text, ContainsSubstring("the revolute joint of body"));
}

TEST_CASE("Assembly: with nothing held the correction is the least one", "[kernel][assembly]")
{
    // All three angles off. The least correction d = q - q_given in the
    // metric M0 (the mass matrix at q_given) on the loop's curve satisfies
    // n . M0 d = 0 for the tangent n of the curve (null space of J) at the
    // solution: M0 d is a combination of the constraint gradients.
    const System sys = four_bar_system();
    const VecX q_given = FourBar::closed(1.0) + Eigen::Vector3d(0.1, 0.25, -0.3);
    Data data(sys.model);
    crba(sys.model, data, q_given);
    const MatX M0 = data.M;

    VecX q = q_given;
    VecX v = VecX::Zero(3);
    const AssemblyReport rep = assemble(sys, q, v, 0.0);
    INFO(rep.summary(sys.model));
    REQUIRE(rep.ok());
    CHECK(rep.position.refinements > 0);
    CHECK(rep.position.freedom_left == 1);
    CHECK(has_code(rep.notes, "MBD-K066"));

    auto optimality = [&](const VecX& at) {
        const MatX J = jacobian(sys, at, 0.0);
        const Eigen::FullPivLU<MatX> lu(J);
        const VecX n = lu.kernel().col(0).normalized();
        const VecX d = at - q_given;   // revolute coordinates: (-) is a subtraction
        return std::abs(n.dot(M0 * d)) / (M0 * d).norm();
    };
    // The refinement stops when the correction moves by less than the
    // tolerance (1e-10 rad); the cosine above is then of that order over |d|
    // (0.4), and the solver's own residual of 1e-10 adds as much.
    const Real at_assembled = optimality(q);
    CHECK(at_assembled <= 1e-8);

    // A plain projection (Gauss-Newton only) is the least correction to first
    // order only: its cosine is of order |d| times the curvature, far above.
    VecX q_gn = q_given, v_gn = VecX::Zero(3);
    ConstraintSolver solver(sys.model, sys.constraints);
    solver.project(data, q_gn, v_gn, 0.0);
    const Real at_projection = optimality(q_gn);
    CHECK(at_projection > 1e3 * at_assembled);

    // Brute force: along the loop, q = closed(th); minimize the M0-norm of
    // q - q_given over th by golden-section search. The minimum is quadratic,
    // so th is found to about sqrt(eps) relative (2e-8), and q to that times
    // |dq/dth| < 2: within 1e-7. The crank (given at 1.1 rad) is the lightest
    // link, so the least correction turns it most: the minimum is near 1.72
    // rad, and the search spans the cost's single valley, 0.5 to 2.5 rad (the
    // crank turns fully: |b - c| < |BD| < b + c for every crank angle).
    auto cost = [&](Real th) {
        const VecX d = FourBar::closed(th) - q_given;
        return d.dot(M0 * d);
    };
    Real lo = 0.5, hi = 2.5;
    const Real g = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int k = 0; k < 200; ++k) {
        const Real x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
        if (cost(x1) < cost(x2)) hi = x2; else lo = x1;
    }
    const VecX q_best = FourBar::closed(0.5 * (lo + hi));
    CHECK((q - q_best).cwiseAbs().maxCoeff() <= 1e-7);
}

TEST_CASE("Assembly: held values that contradict the constraints are reported", "[kernel][assembly]")
{
    const System sys = four_bar_system();

    SECTION("positions")
    {
        // Crank and rocker angles held, the rocker 0.2 rad away from where the
        // loop puts it: one coordinate is left for two independent equations.
        VecX q = FourBar::closed(1.0) + Eigen::Vector3d(0.0, 0.0, 0.2);
        VecX v = VecX::Zero(3);
        const VecX q_given = q;
        AssemblySpec spec;
        spec.hold_positions = {0, 2};
        const AssemblyReport rep = assemble(sys, q, v, 0.0, spec);
        INFO(rep.summary(sys.model));
        CHECK_FALSE(rep.ok());
        CHECK(has_code(rep.errors, "MBD-K062"));
        CHECK(has_code(rep.warnings, "MBD-K065"));   // 2 - 1 = 1 direction taken away
        CHECK(q(0) == q_given(0));
        CHECK(q(2) == q_given(2));
        CHECK(rep.position.residual < rep.position.residual_before);   // the best it could do
        CHECK_FALSE(rep.velocity.done);
        CHECK_THAT(rep.summary(sys.model), ContainsSubstring("NOT CONVERGED"));
    }

    SECTION("positions held consistently")
    {
        // Every angle held, at the closure: nothing to do, but the warning says
        // that the held values carry the constraints by themselves.
        VecX q = FourBar::closed(1.0);
        VecX v = VecX::Zero(3);
        AssemblySpec spec;
        spec.hold_positions = {0, 1, 2};
        const AssemblyReport rep = assemble(sys, q, v, 0.0, spec);
        CHECK(rep.ok());
        CHECK(has_code(rep.warnings, "MBD-K065"));
        CHECK(rep.position.largest_change_at == -1);
    }

    SECTION("velocities")
    {
        // The crank turns, and the coupler's relative rate is held at zero:
        // the loop's single motion cannot do that.
        VecX q = FourBar::closed(1.0);
        VecX v = Eigen::Vector3d(2.0, 0.0, 0.0);
        AssemblySpec spec;
        spec.hold_velocities = {0, 1};
        const AssemblyReport rep = assemble(sys, q, v, 0.0, spec);
        CHECK_FALSE(rep.ok());
        CHECK(has_code(rep.errors, "MBD-K063"));
        CHECK(rep.position.converged);
        CHECK(v(0) == 2.0);
        CHECK(v(1) == 0.0);
    }
}

TEST_CASE("Assembly: a rotation held whole is kept exactly", "[kernel][assembly]")
{
    // A free body whose point r is tied to the ground point P by a spherical
    // closure, given 5 cm away. With its rotation held, the translation that
    // closes the joint is unique: P - R r.
    System sys;
    const RigidBodyInertia I = RigidBodyInertia::from_solid_box(2.0, Vec3(0.3, 0.2, 0.1));
    const int body = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(),
                                        Transform3(), I, "floater");
    const Vec3 P(0.4, -0.2, 0.7), r(0.1, 0.25, -0.15);
    sys.constraints.push_back(spherical_closure(Marker{0, Transform3::FromTranslation(P)},
                                                Marker{body, Transform3::FromTranslation(r)}));
    const Quat Q = Quat(Eigen::AngleAxisd(0.7, Vec3(1.0, 2.0, -0.5).normalized()));
    const Vec3 t_exact = P - Q * r;
    VecX q(7);
    q << t_exact + Vec3(0.03, -0.04, 0.0), Q.coeffs();
    VecX v = VecX::Zero(6);

    SECTION("rotation held")
    {
        AssemblySpec spec;
        spec.hold_positions = {0, 1, 2};
        const AssemblyReport rep = assemble(sys, q, v, 0.0, spec);
        CHECK(rep.ok());
        // integrate() moves the rotation by exp(0) = identity: only the
        // final normalization may touch the quaternion, by an ulp or so.
        CHECK((q.tail<4>() - Q.coeffs()).cwiseAbs().maxCoeff() <= 1e-15);
        CHECK((q.head<3>() - t_exact).cwiseAbs().maxCoeff() <= 1e-12);
        CHECK(rep.position.freedom_left == 0);
    }

    SECTION("nothing held")
    {
        const AssemblyReport rep = assemble(sys, q, v, 0.0);
        CHECK(rep.ok());
        CHECK(phi_norm(sys, q, 0.0) <= 1e-10);
        CHECK(rep.position.freedom_left == 3);
        CHECK(has_code(rep.notes, "MBD-K066"));
    }
}

TEST_CASE("Assembly: malformed hold lists are refused", "[kernel][assembly]")
{
    const System four = four_bar_system();
    VecX q = FourBar::closed(1.0), v = VecX::Zero(3);
    AssemblySpec spec;
    spec.hold_positions = {3};
    CHECK_THROWS_WITH(assemble(four, q, v, 0.0, spec), ContainsSubstring("MBD-K060"));
    CHECK_THROWS_WITH(AssemblySpec().hold(four.model, 0), ContainsSubstring("MBD-K060"));
    CHECK_THROWS_WITH(AssemblySpec().hold(four.model, 1, 1), ContainsSubstring("MBD-K060"));

    // Part of a rotation.
    System sys;
    const RigidBodyInertia I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.2, 0.3));
    const int ball = sys.model.add_body(0, std::make_shared<SphericalJointModel>(), Transform3(), Transform3(), I);
    const int floating = sys.model.add_body(ball, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), I);
    VecX q2 = sys.model.neutral_configuration(), v2 = VecX::Zero(sys.model.nv);
    const int f = sys.model.idx_v[static_cast<std::size_t>(floating)];

    AssemblySpec part;
    part.hold(sys.model, ball, 0);
    CHECK_THROWS_WITH(assemble(sys, q2, v2, 0.0, part), ContainsSubstring("MBD-K061"));

    AssemblySpec linear_part;
    linear_part.hold_positions = {f + 3};   // translation in part, rotation free
    CHECK_THROWS_WITH(assemble(sys, q2, v2, 0.0, linear_part), ContainsSubstring("MBD-K061"));

    AssemblySpec allowed;
    allowed.hold_positions = {f, f + 1, f + 2, f + 5};   // rotation whole, then any translation
    allowed.hold(sys.model, ball);
    CHECK_NOTHROW(assemble(sys, q2, v2, 0.0, allowed));
}

TEST_CASE("Assembly: a simulator is assembled at its time, drivers included", "[kernel][assembly]")
{
    // A pendulum driven along s(t) = A sin(w t). Assembled at t = 0.3 s from
    // rest at zero, it must sit at s(t) and move at s'(t): the driver is a
    // constraint like any other.
    const Real A = 0.4, w = 3.0, t = 0.3;
    System sys;
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.5));
    I.com_B = Vec3(0.5, 0.0, 0.0);
    const int body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), I);
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, body,
        TimeFunction([=](Real tt) { return A * std::sin(w * tt); },
                     [=](Real tt) { return A * w * std::cos(w * tt); },
                     [=](Real tt) { return -A * w * w * std::sin(w * tt); })));
    Simulator sim(sys);
    sim.time = t;
    const AssemblyReport rep = assemble(sim);
    CHECK(rep.ok());
    CHECK(std::abs(sim.q(0) - A * std::sin(w * t)) <= 1e-12);
    CHECK(std::abs(sim.v(0) - A * w * std::cos(w * t)) <= 1e-12);
    CHECK(rep.degrees_of_freedom == 0);
    // The kinematics were refreshed: the body is turned about Z by the
    // assembled angle, so its quaternion is (0, 0, sin(q/2), cos(q/2)).
    const Real angle = 2.0 * std::atan2(sim.states()[static_cast<std::size_t>(body)].q_WB.z(),
                                        sim.states()[static_cast<std::size_t>(body)].q_WB.w());
    CHECK(std::abs(angle - sim.q(0)) <= 1e-12);

    // Holding the driven coordinate at another value cannot succeed, and the
    // simulator says so through the diagnostic sink.
    captured_warnings().clear();
    const DiagnosticSink previous = diagnostic_sink();
    diagnostic_sink() = [](const std::string& m) { captured_warnings().push_back(m); };
    sim.q(0) = 0.0;
    AssemblySpec spec;
    spec.hold(sys.model, body);
    const AssemblyReport bad = assemble(sim, spec);
    diagnostic_sink() = previous;
    CHECK_FALSE(bad.ok());
    REQUIRE(captured_warnings().size() == 1);
    CHECK_THAT(captured_warnings()[0], ContainsSubstring("MBD-K067"));
    CHECK_THAT(captured_warnings()[0], ContainsSubstring("MBD-K062"));
}
