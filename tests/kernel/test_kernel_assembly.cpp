// Initial-condition assembly (plan task 3.4).
//
//   - a four-bar given an open loop and velocities that would stretch it is
//     assembled with its crank held, to the closed-form configuration and
//     loop velocities, and reported;
//   - with nothing held, the correction is the smallest one: it meets the
//     optimality condition, no configuration of the mechanism is closer, and
//     Gauss-Newton alone (the first stage) does not find it;
//   - a coordinate that no constraint involves is never changed;
//   - held coordinates that over-determine the loop are reported;
//   - the held parts of a free joint are kept exactly;
//   - a hold on a coordinate that does not exist is refused;
//   - constraints that contradict each other are named (task 5.2, D44): of a
//     model with a four-bar that closes and a bob held at 1 m from two
//     anchors 3 m apart, the two distances, and nothing else.

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

/// The diagonal of the mass matrix at q: assemble()'s weights.
VecX mass_diagonal(const Model& model, const VecX& q)
{
    Data data(model);
    return crba(model, data, q).diagonal();
}

/// |n^T W d| / (|n|_W |d|_W): zero when the correction d is perpendicular, in
/// the metric W, to the mechanism's motion n, the condition for the least
/// correction on a mechanism with one degree of freedom.
Real optimality_cosine(const VecX& n, const VecX& d, const VecX& w)
{
    const auto W = w.asDiagonal();
    return std::abs(n.dot(W * d)) / std::sqrt(n.dot(W * n) * d.dot(W * d));
}

bool has_warning(const AssemblyReport& r, const char* code)
{
    for (const auto& m : r.warnings) {
        if (m.find(code) != std::string::npos) return true;
    }
    return false;
}

} // namespace

TEST_CASE("Assembly: a four-bar given inconsistent values is assembled and reported",
          "[kernel][assembly]")
{
    // The loop is open by a few centimetres and the velocities would stretch
    // it. The crank is held at 1 rad turning at 2 rad/s: with it held the
    // loop has no freedom left, so the result is the closed-form
    // configuration at that crank angle, and the loop velocities for that
    // crank rate.
    FourBar fb;
    const Model& model = fb.sys.model;
    Simulator sim(fb.sys);
    const Eigen::Vector3d closed = FourBar::configuration(1.0);
    sim.q = closed + Eigen::Vector3d(0.0, 0.05, -0.04);
    sim.v = Eigen::Vector3d(2.0, 0.3, -0.5);
    AssemblyOptions options;
    options.hold_joint(model, fb.crank);

    const AssemblyReport r = assemble(sim, options);
    CHECK(r.ok());
    CHECK(r.smallest_correction);
    CHECK(r.warnings.empty());
    CHECK(r.position_residual_given > 0.01);
    CHECK(r.velocity_residual_given > 0.1);

    // The held crank exactly as given; the rest closed.
    CHECK(sim.q(0) == 1.0);
    CHECK(sim.v(0) == 2.0);
    CHECK((sim.q - closed).norm() < 1e-10);
    const Eigen::Vector3d v_closed = 2.0 * FourBar::motion(sim.q);
    CHECK((sim.v - v_closed).norm() < 1e-10);
    CHECK(r.position_residual < 1e-10);
    CHECK(r.velocity_residual < 1e-10);

    // The planar loop closed in 3D: 5 equations of rank 2, one degree of
    // freedom, taken by the crank.
    CHECK(r.equations == 5);
    CHECK(r.rank == 2);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(r.position_hold_rank == 2);
    CHECK(r.velocity_hold_rank == 2);

    // Accelerations: J a = gamma, and the power of the forces of the motion
    // vanishes, v^T (M a + b) = v^T J^T lambda = (J v)^T lambda = 0. With one
    // degree of freedom, along v, these two determine a.
    CHECK(r.acceleration_residual < 1e-9);
    Data data(model);
    const VecX f = rnea(model, data, sim.q, sim.v, r.loads.v_dot);
    CHECK(std::abs(sim.v.dot(f)) < 1e-10 * sim.v.norm() * f.norm());
    CHECK(f.norm() > 1.0);   // the loop does carry forces

    // The report names every coordinate.
    {
        const std::string s = r.summary(model);
        INFO(s);
        CHECK_THAT(s, ContainsSubstring("assembled"));
        CHECK_THAT(s, ContainsSubstring("5 equations of rank 2; velocities 3; degrees of freedom 1"));
        CHECK_THAT(s, ContainsSubstring("1 in position, 1 in velocity; degrees of freedom left 0"));
        CHECK_THAT(s, ContainsSubstring("[0] body 1 (crank), revolute: position 1 (held); velocity 2 (held)"));
        CHECK_THAT(s, ContainsSubstring("[1] body 2 (coupler), revolute: position "));
        CHECK_THAT(s, ContainsSubstring("[2] body 3 (rocker), revolute: position "));
    }
}

TEST_CASE("Assembly: with nothing held the correction is the smallest one", "[kernel][assembly]")
{
    // Every coordinate off by several hundredths of a radian. The assembled
    // configuration must be the point of the mechanism's one-dimensional set
    // of configurations nearest the given one, in the metric of the
    // diagonal of the mass matrix there.
    FourBar fb;
    const Model& model = fb.sys.model;
    Simulator sim(fb.sys);
    const VecX given = FourBar::configuration(1.0) + Eigen::Vector3d(0.08, -0.06, 0.05);
    sim.q = given;
    const AssemblyReport r = assemble(sim);
    REQUIRE(r.ok());
    CHECK(r.smallest_correction);
    const VecX w = mass_diagonal(model, given);

    // The optimality condition: the correction is perpendicular, in that
    // metric, to the motion the mechanism allows at the assembled point.
    const Real cos_assembled = optimality_cosine(FourBar::motion(sim.q), r.position_change, w);
    CHECK(cos_assembled < 1e-8);

    // No configuration is closer: a golden-section search along the
    // mechanism's configurations, parametrized by the crank angle.
    auto distance2 = [&](Real th1) {
        const VecX dq = FourBar::configuration(th1) - given;
        return dq.dot(w.asDiagonal() * dq);
    };
    Real lo = sim.q(0) - 0.2, hi = sim.q(0) + 0.2;
    const Real golden = 0.5 * (std::sqrt(5.0) - 1.0);
    while (hi - lo > 1e-11) {
        const Real x1 = hi - golden * (hi - lo), x2 = lo + golden * (hi - lo);
        if (distance2(x1) < distance2(x2)) hi = x2;
        else lo = x1;
    }
    const Real th_best = 0.5 * (lo + hi);
    CHECK(std::abs(th_best - sim.q(0)) < 1e-6);
    CHECK(distance2(sim.q(0)) <= distance2(th_best) * (1.0 + 1e-12));

    // The first stage alone, Gauss-Newton onto the constraints with the same
    // weights, lands on the constraints but not at the nearest point: each
    // of its steps is the least from where it starts, not from the given
    // values. Here on the planar loop equations: dq = -W^-1 J^T (J W^-1
    // J^T)^-1 gap.
    VecX q = given;
    const Eigen::Vector3d w_inv = w.cwiseInverse();
    Eigen::Matrix<Real, 2, 3> J;
    for (int k = 0; k < 20 && FourBar::gap(q, &J).norm() > 1e-14; ++k) {
        const Eigen::Matrix2d A = J * w_inv.asDiagonal() * J.transpose();
        q -= w_inv.asDiagonal() * J.transpose() * (A.inverse() * FourBar::gap(q));
    }
    // The kernel's five closure equations agree that the loop is closed.
    ConstraintSolver solver(model, fb.sys.constraints);
    Data data(model);
    solver.evaluate(data, q, VecX::Zero(3), 0.0);
    REQUIRE(solver.phi().norm() < 1e-12);
    const Real cos_projected = optimality_cosine(FourBar::motion(q), q - given, w);
    CHECK(cos_projected > 1e-4);
    CHECK(distance2(q(0)) > distance2(sim.q(0)));
    INFO("optimality cosine: assembled " << cos_assembled << ", Gauss-Newton alone " << cos_projected
         << "; iterations " << r.projection_iterations << " + " << r.refinement_iterations
         << "; distance^2 " << distance2(sim.q(0)) << " against " << distance2(q(0)));
    CHECK(r.refinement_iterations <= 20);
}

TEST_CASE("Assembly: a coordinate that no constraint involves is not changed", "[kernel][assembly]")
{
    // A pendulum hangs from the coupler, outside the loop. Its coordinate
    // and rate stay as given, because the weights are diagonal (D25). The
    // full mass matrix, the metric of ConstraintSolver::project, couples the
    // pendulum to the coupler (it hangs from it) and would turn it too.
    FourBar fb(true);
    const Model& model = fb.sys.model;
    Simulator sim(fb.sys);
    VecX given(4), v_given(4);
    given << FourBar::configuration(1.0) + Eigen::Vector3d(0.04, -0.03, 0.02), 0.3;
    v_given << 1.0, -2.0, 0.5, 0.7;
    sim.q = given;
    sim.v = v_given;
    const AssemblyReport r = assemble(sim);
    REQUIRE(r.ok());
    CHECK(sim.q(3) == given(3));
    CHECK(sim.v(3) == v_given(3));
    CHECK(r.position_residual_given > 1e-3);   // the loop was open,
    CHECK(r.position_residual < 1e-10);        // and was closed
    CHECK(r.velocity_residual_given > 0.1);

    ConstraintSolver solver(model, fb.sys.constraints);
    Data data(model);
    VecX q = given, v = v_given;
    REQUIRE(solver.project(data, q, v, 0.0).converged);
    CHECK(std::abs(q(3) - given(3)) > 1e-4);
    CHECK(std::abs(v(3) - v_given(3)) > 1e-3);
}

TEST_CASE("Assembly: held coordinates that over-determine the loop are reported", "[kernel][assembly]")
{
    // Crank and rocker held: only the coupler can move, one coordinate for
    // two independent equations.
    FourBar fb;
    Simulator sim(fb.sys);
    const VecX closed = FourBar::configuration(1.0);
    AssemblyOptions options;
    options.hold_position = {0, 2};

    SECTION("inconsistent held values: not assembled")
    {
        const VecX given = closed + Eigen::Vector3d(0.0, 0.02, 0.05);
        sim.q = given;
        const AssemblyReport r = assemble(sim, options);
        INFO(r.summary(fb.sys.model));
        CHECK(!r.ok());
        CHECK(!r.positions_assembled);
        CHECK(r.rank == 2);
        CHECK(r.position_hold_rank == 1);
        CHECK(has_warning(r, "MBD-K062"));
        CHECK(has_warning(r, "MBD-K063"));
        CHECK(sim.q(0) == given(0));
        CHECK(sim.q(2) == given(2));
        CHECK_THAT(r.summary(fb.sys.model), ContainsSubstring("NOT assembled"));
    }
    SECTION("consistent held values: assembled, the redundancy still reported")
    {
        const VecX given = closed + Eigen::Vector3d(0.0, 0.03, 0.0);
        sim.q = given;
        const AssemblyReport r = assemble(sim, options);
        INFO(r.summary(fb.sys.model));
        CHECK(r.ok());
        CHECK((sim.q - closed).norm() < 1e-10);
        CHECK(!has_warning(r, "MBD-K062"));
        CHECK(has_warning(r, "MBD-K063"));
    }
}

TEST_CASE("Assembly: the held parts of a free joint are kept", "[kernel][assembly]")
{
    // A body on a free joint (its centre of mass at its origin, the same
    // inertia about every axis), a point of it at r held at the ground point
    // G by a point coincidence.
    System sys;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(2.0, Vec3(0.1, 0.1, 0.1)), "probe");
    const Vec3 r_B(0.4, 0.0, 0.0), G(0.0, 0.0, 1.0);
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{0, Transform3::FromTranslation(G)}, Marker{1, Transform3::FromTranslation(r_B)}));
    Simulator sim(sys);
    const Quat R0(Eigen::AngleAxisd(0.5, Vec3(1.0, -2.0, 0.5).normalized()));
    auto place = [&](const Vec3& p) {
        sim.q.head<3>() = p;
        sim.q.tail<4>() = R0.coeffs();
    };

    SECTION("orientation held: the body translates, to p = G - R0 r")
    {
        place(Vec3(0.1, 0.2, 1.3));
        AssemblyOptions options;
        options.hold_position = {0, 1, 2};
        const AssemblyReport r = assemble(sim, options);
        REQUIRE(r.ok());
        CHECK(r.smallest_correction);
        CHECK(r.warnings.empty());
        const Vec3 p = sim.q.head<3>();
        CHECK((p - (G - R0 * r_B)).norm() < 1e-12);
        CHECK((sim.q.tail<4>() - R0.coeffs()).norm() < 1e-15);
    }
    SECTION("position held: the body turns, by the smallest rotation")
    {
        // The origin a distance |r| from G, so that a rotation alone closes
        // the gap. With equal weights on the three rotations, the smallest
        // is the one about the axis across R0 r and u = G - p, by the angle
        // between them.
        const Vec3 u = 0.4 * Vec3(1.0, 2.0, 2.0) / 3.0;
        const Vec3 p0 = G - u;
        place(p0);
        AssemblyOptions options;
        options.hold_position = {3, 4, 5};
        const AssemblyReport r = assemble(sim, options);
        REQUIRE(r.ok());
        CHECK(r.smallest_correction);
        // A rotation moves the point only across r, so with the position
        // held the three equations have rank 2: the distance |G - p| = |r|
        // is fixed by the held values. They agree here, so it is assembled,
        // and the dependence is reported.
        CHECK(r.rank == 3);
        CHECK(r.position_hold_rank == 2);
        CHECK(has_warning(r, "MBD-K063"));
        CHECK(sim.q(0) == p0.x());
        CHECK(sim.q(1) == p0.y());
        CHECK(sim.q(2) == p0.z());
        const Vec3 a = R0 * r_B;
        const Quat R_expected = Quat(Eigen::AngleAxisd(std::acos(a.normalized().dot(u.normalized())),
                                                       a.cross(u).normalized())) * R0;
        const Quat R(sim.q(6), sim.q(3), sim.q(4), sim.q(5));
        CHECK(R.angularDistance(R_expected) < 1e-9);
    }
}

TEST_CASE("Assembly: a hold on a coordinate that does not exist is refused", "[kernel][assembly]")
{
    FourBar fb;
    Simulator sim(fb.sys);
    AssemblyOptions options;
    options.hold_position = {3};
    CHECK_THROWS_WITH(assemble(sim, options), ContainsSubstring("MBD-K060"));
    CHECK_THROWS_WITH(AssemblyOptions().hold_joint(fb.sys.model, 0), ContainsSubstring("MBD-K060"));
    CHECK_THROWS_WITH(AssemblyOptions().hold_joint(fb.sys.model, 4), ContainsSubstring("MBD-K060"));
}

TEST_CASE("Assembly: constraints that contradict each other are named, and only they", "[kernel][assembly]")
{
    // The four-bar closes (its revolute closure, constraint 0). The bob, a
    // free body, must be 1 m from (0, 0, 2) and 1 m from (3, 0, 2): no point
    // is, so the positions cannot be assembled. Gauss-Newton stops between
    // the anchors, where the two distance equations, linearised, ask for
    // opposite moves along X (each alone could be met): those two equations,
    // constraints 1 and 2, are the smallest set that cannot all hold, and no
    // equation of the four-bar belongs to it.
    FourBar fb;
    const int bob = fb.sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                          RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)), "bob");
    fb.sys.constraints.push_back(
        std::make_shared<Distance>(Marker{0, Transform3::FromTranslation(Vec3(0.0, 0.0, 2.0))}, Marker{bob, Transform3()}, 1.0));
    fb.sys.constraints.push_back(
        std::make_shared<Distance>(Marker{0, Transform3::FromTranslation(Vec3(3.0, 0.0, 2.0))}, Marker{bob, Transform3()}, 1.0));
    Simulator sim(fb.sys);
    sim.q.head(3) = FourBar::configuration(0.7);
    sim.q.segment<3>(3) = Vec3(1.2, 0.3, 2.1);   // the bob, somewhere between the anchors
    const AssemblyReport r = assemble(sim);
    INFO(r.summary(fb.sys.model));
    REQUIRE_FALSE(r.positions_assembled);
    const int first = fb.sys.constraints[0]->size();   // the distances' rows follow the closure's
    CHECK(r.conflicting_equations == std::vector<int>{first, first + 1});
    REQUIRE(has_warning(r, "MBD-K062"));
    for (const auto& m : r.warnings) {
        if (m.find("MBD-K062") == std::string::npos) continue;
        CHECK_THAT(m, ContainsSubstring("constraint 1 (distance), its equation 1 of 1, on body 0 (ground) and body 4 (bob)"));
        CHECK_THAT(m, ContainsSubstring("constraint 2 (distance)"));
        CHECK(m.find("constraint 0") == std::string::npos);
    }
}

TEST_CASE("Assembly: a loop far from closing but not contradicted says so", "[kernel][assembly]")
{
    // The four-bar alone, given positions that need more Gauss-Newton steps
    // than allowed (one): the linearised equations there can all hold, so no
    // equation is named, and the message says why the positions were not
    // assembled instead.
    FourBar fb;
    Simulator sim(fb.sys);
    sim.q.head(3) = FourBar::configuration(0.7) + Vec3(0.4, -0.3, 0.2);
    AssemblyOptions options;
    options.max_iterations = 1;
    const AssemblyReport r = assemble(sim, options);
    INFO(r.summary(fb.sys.model));
    REQUIRE_FALSE(r.positions_assembled);
    CHECK(r.conflicting_equations.empty());
    bool said = false;
    for (const auto& m : r.warnings) {
        if (m.find("MBD-K062") != std::string::npos) said = m.find("can all hold there") != std::string::npos;
    }
    CHECK(said);
}
