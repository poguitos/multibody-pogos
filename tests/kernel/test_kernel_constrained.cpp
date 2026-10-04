// Constrained dynamics on the kernel (plan task 2.6).
//
//   - a body held by a closure moves, and is held, exactly like the same body
//     on the corresponding tree joint: same accelerations, and constraint
//     forces equal to the tree's joint forces;
//   - redundant constraints are detected and do not change the motion;
//   - a planar four-bar, closed in 3D by a revolute closure (5 equations of
//     which 2 are independent), conserves energy over two seconds;
//   - projection returns a perturbed state onto the constraints;
//   - nothing is allocated after setup (hidden test, run by CI).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#include <cstdlib>
#endif

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::Rng;

namespace {

using Closure = std::function<std::shared_ptr<ConstraintSet>(Marker, Marker)>;

/// The free-joint state of a body placed at oMi with body velocity v_B.
void free_state(const Transform3& oMi, const Vec6& v_B, VecX& q, VecX& v)
{
    q.resize(7);
    q << oMi.p, oMi.q.coeffs();
    v = v_B;
}

/// A planar four-bar in the XY plane, hinges along Z: crank AB, coupler BC,
/// rocker CD, ground AD. A tree of three revolute joints, closed at D.
struct FourBar {
    static constexpr Real a = 0.3, b = 0.8, c = 0.6, d = 0.7;
    Model model;
    std::vector<std::shared_ptr<const ConstraintModel>> closure;
    VecX q;

    FourBar()
    {
        model.gravity = Vec3(0.0, -g_accel, 0.0);
        const auto hinge = std::make_shared<RevoluteJointModel>();
        auto link = [](Real length, Real mass) {
            RigidBodyInertia I = RigidBodyInertia::from_solid_box(mass, Vec3(0.5 * length, 0.02, 0.02));
            I.com_B = Vec3(0.5 * length, 0.0, 0.0);
            return I;
        };
        const Transform3 I3;
        const int crank   = model.add_body(0, hinge, I3, I3, link(a, 1.0), "crank");
        const int coupler = model.add_body(crank, hinge, Transform3::FromTranslation(Vec3(a, 0, 0)),
                                           I3, link(b, 2.0), "coupler");
        const int rocker  = model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(b, 0, 0)),
                                           I3, link(c, 1.5), "rocker");
        closure.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(d, 0, 0))},
                                           Marker{rocker, Transform3::FromTranslation(Vec3(c, 0, 0))}));

        // Assemble at crank angle 1 rad: C is where the circles about B
        // (radius b) and D (radius c) meet, on the upper side.
        const Real th1 = 1.0;
        const Eigen::Vector2d B(a * std::cos(th1), a * std::sin(th1)), D(d, 0.0);
        const Real dist = (D - B).norm();
        const Real along = (b * b - c * c + dist * dist) / (2.0 * dist);
        const Real h = std::sqrt(b * b - along * along);
        const Eigen::Vector2d e = (D - B) / dist;
        const Eigen::Vector2d C = B + along * e + h * Eigen::Vector2d(-e.y(), e.x());
        const Real phi2 = std::atan2(C.y() - B.y(), C.x() - B.x());
        const Real phi3 = std::atan2(D.y() - C.y(), D.x() - C.x());
        q.resize(3);
        q << th1, phi2 - th1, phi3 - phi2;
    }
};

Real energy(const Model& model, Data& data, const VecX& q, const VecX& v)
{
    forward_kinematics(model, data, q, v);
    return kinetic_energy(model, data) + potential_energy(model, data);
}

} // namespace

TEST_CASE("Kernel constrained dynamics: a closure acts like the tree joint it stands for",
          "[kernel][constrained]")
{
    struct Case {
        const char* name;
        std::shared_ptr<const JointModel> joint;
        Closure closure;
        Transform3 turn;  // marker j relative to the child-side joint frame
    };
    const Transform3 none;
    const std::vector<Case> cases{
        {"spherical", std::make_shared<SphericalJointModel>(), spherical_closure, none},
        {"revolute", std::make_shared<RevoluteJointModel>(), revolute_closure, none},
        // The tree joint's second axis is X of the child-side frame; the
        // closure wants the arms of the cross along Z of each marker.
        {"universal", std::make_shared<UniversalJointModel>(), universal_closure,
         Transform3(Quat(Eigen::AngleAxisd(0.5 * pi, Vec3::UnitY())), Vec3::Zero())},
        {"cylindrical", std::make_shared<CylindricalJointModel>(), cylindrical_closure, none},
        {"prismatic", std::make_shared<PrismaticJointModel>(), prismatic_closure, none},
    };

    Rng rng(51);
    for (const Case& k : cases) {
        INFO(k.name);
        const Transform3 X_PJ = rng.frame(0.5), X_CJ = rng.frame(0.3);
        const RigidBodyInertia inertia = mbd_test::random_inertia(rng);

        Model tree;
        tree.add_body(0, k.joint, X_PJ, X_CJ, inertia);
        Model loose;
        loose.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), inertia);
        ConstraintSolver solver(loose, {k.closure(Marker{0, X_PJ}, Marker{1, X_CJ * k.turn})});

        // The same state in both.
        const VecX q_t = mbd_test::random_configuration(tree, rng);
        const VecX v_t = mbd_test::random_vector(tree.nv, rng, 1.5);
        Data d_t(tree), d_l(loose);
        forward_kinematics(tree, d_t, q_t, v_t);
        VecX q_l, v_l;
        free_state(d_t.oMi[1], d_t.v[1], q_l, v_l);

        // Passive joint under gravity.
        const VecX a_t = aba(tree, d_t, q_t, v_t, VecX::Zero(tree.nv));
        const Vec6 A_t = body_acceleration_world(d_t, 1);
        const VecX a_l = solver.forward_dynamics(d_l, q_l, v_l, VecX::Zero(6), 0.0);
        CHECK(solver.phi().norm() < 1e-12);
        CHECK(solver.info().rank == solver.size());
        CHECK(!solver.info().redundant());

        forward_kinematics(loose, d_l, q_l, v_l, a_l);
        const Vec6 A_l = body_acceleration_world(d_l, 1);
        CHECK((A_l - A_t).norm() < 1e-9 * (1.0 + A_t.norm()));

        // The constraint force on the body is the tree's joint force.
        rnea(tree, d_t, q_t, v_t, a_t);
        const Vec6 f_joint = d_t.f[1];
        const Vec6 f_constraint = solver.J().transpose() * solver.lambda();
        CHECK((f_constraint - f_joint).norm() < 1e-8 * (1.0 + f_joint.norm()));
    }
}

TEST_CASE("Kernel constrained dynamics: redundant constraints do not change the motion",
          "[kernel][constrained]")
{
    Rng rng(52);
    const Transform3 X_PJ = rng.frame(0.5), X_CJ = rng.frame(0.3);
    Model model;
    model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                   mbd_test::random_inertia(rng));
    const Marker ground{0, X_PJ}, body{1, X_CJ};
    ConstraintSolver once(model, {revolute_closure(ground, body)});
    ConstraintSolver twice(model, {revolute_closure(ground, body), revolute_closure(ground, body)});

    // An assembled state: the body on the hinge, turning about it.
    Model hinge;
    hinge.add_body(0, std::make_shared<RevoluteJointModel>(), X_PJ, X_CJ, model.inertia[1]);
    Data d_h(hinge), data(model);
    VecX q_h(1), v_h(1);
    q_h << 0.7;
    v_h << -1.3;
    forward_kinematics(hinge, d_h, q_h, v_h);
    VecX q, v;
    free_state(d_h.oMi[1], d_h.v[1], q, v);

    const VecX tau = mbd_test::random_vector(6, rng, 3.0);
    const VecX a1 = once.forward_dynamics(data, q, v, tau, 0.0);
    const Vec6 f1 = once.J().transpose() * once.lambda();
    const VecX a2 = twice.forward_dynamics(data, q, v, tau, 0.0);
    const Vec6 f2 = twice.J().transpose() * twice.lambda();

    CHECK(once.info().equations == 5);
    CHECK(once.info().rank == 5);
    CHECK(twice.info().equations == 10);
    CHECK(twice.info().rank == 5);
    CHECK(twice.info().redundant());
    CHECK((a2 - a1).norm() < 1e-9 * (1.0 + a1.norm()));
    CHECK((f2 - f1).norm() < 1e-8 * (1.0 + f1.norm()));
}

TEST_CASE("Kernel constrained dynamics: a four-bar linkage conserves energy",
          "[kernel][constrained]")
{
    FourBar fb;
    const Model& model = fb.model;
    Data data(model);
    ConstraintSolver solver(model, fb.closure);

    // Start the crank at 2 rad/s and make the velocities consistent.
    VecX q = fb.q, v(3);
    v << 2.0, 0.0, 0.0;
    const ProjectionInfo start = solver.project(data, q, v, 0.0);
    CHECK(start.converged);
    CHECK(start.position_residual < 1e-12);
    CHECK(start.velocity_residual < 1e-12);

    // The planar loop closed in 3D: 5 equations, 2 independent, 1 freedom.
    solver.forward_dynamics(data, q, v, VecX::Zero(3), 0.0);
    CHECK(solver.info().equations == 5);
    CHECK(solver.info().rank == 2);

    const Real E0 = energy(model, data, q, v);
    const VecX zero = VecX::Zero(3);
    const Real dt = 1e-3;
    Real crank_min = q(0), crank_max = q(0);
    for (int step = 0; step < 2000; ++step) {
        // RK4; every joint is revolute, so q_dot = v.
        const VecX k1v = solver.forward_dynamics(data, q, v, zero, 0.0);
        const VecX k1q = v;
        const VecX q2 = q + 0.5 * dt * k1q, v2 = v + 0.5 * dt * k1v;
        const VecX k2v = solver.forward_dynamics(data, q2, v2, zero, 0.0);
        const VecX q3 = q + 0.5 * dt * v2, v3 = v + 0.5 * dt * k2v;
        const VecX k3v = solver.forward_dynamics(data, q3, v3, zero, 0.0);
        const VecX q4 = q + dt * v3, v4 = v + dt * k3v;
        const VecX k4v = solver.forward_dynamics(data, q4, v4, zero, 0.0);
        q += dt / 6.0 * (k1q + 2.0 * v2 + 2.0 * v3 + v4);
        v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        const ProjectionInfo p = solver.project(data, q, v, 0.0);
        REQUIRE(p.converged);
        crank_min = std::min(crank_min, q(0));
        crank_max = std::max(crank_max, q(0));
    }
    const Real E1 = energy(model, data, q, v);
    CHECK(std::abs(E1 - E0) < 1e-7 * (1.0 + std::abs(E0)));

    // The linkage really moved: the crank turns right round (about twice).
    CHECK(crank_max - crank_min > 2.0 * pi);
}

TEST_CASE("Kernel constrained dynamics: projection returns a perturbed state to the constraints",
          "[kernel][constrained]")
{
    FourBar fb;
    Data data(fb.model);
    ConstraintSolver solver(fb.model, fb.closure);
    VecX q = fb.q + Eigen::Vector3d(0.05, -0.04, 0.03);
    VecX v = Eigen::Vector3d(1.0, -2.0, 0.5);

    solver.evaluate(data, q, v, 0.0);
    CHECK(solver.phi().norm() > 5e-3);   // the loop is open by about a centimetre
    const ProjectionInfo p = solver.project(data, q, v, 0.0);
    CHECK(p.converged);
    CHECK(p.iterations <= 4);            // Gauss-Newton converges quadratically
    CHECK(p.position_residual < 1e-10);
    CHECK(p.velocity_residual < 1e-10);
}

TEST_CASE("Kernel constrained dynamics: Baumgarte stabilization pulls the state back",
          "[kernel][constrained]")
{
    // Started off the constraints at rest and integrated without projection:
    // with stabilization the error obeys phi'' + 2 alpha phi' + beta^2 phi = 0 and
    // decays (critically damped for alpha = beta); without it phi'' = 0, so
    // it stays where it started.
    FourBar fb;
    Data data(fb.model);
    const VecX q_bad = fb.q + Eigen::Vector3d(0.0, 0.02, 0.0);

    auto final_error = [&](Real alpha, Real beta) {
        ConstraintSolver solver(fb.model, fb.closure);
        solver.baumgarte_alpha = alpha;
        solver.baumgarte_beta  = beta;
        VecX q = q_bad, v = VecX::Zero(3);
        const VecX zero = VecX::Zero(3);
        const Real dt = 1e-3;
        for (int step = 0; step < 2000; ++step) {
            // RK4; q_dot = v for revolute joints.
            const VecX k1 = solver.forward_dynamics(data, q, v, zero, 0.0);
            const VecX v2 = v + 0.5 * dt * k1;
            const VecX k2 = solver.forward_dynamics(data, VecX(q + 0.5 * dt * v), v2, zero, 0.0);
            const VecX v3 = v + 0.5 * dt * k2;
            const VecX k3 = solver.forward_dynamics(data, VecX(q + 0.5 * dt * v2), v3, zero, 0.0);
            const VecX v4 = v + dt * k3;
            const VecX k4 = solver.forward_dynamics(data, VecX(q + dt * v3), v4, zero, 0.0);
            q += dt / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4);
            v += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        solver.evaluate(data, q, v, 0.0);
        return solver.phi().norm();
    };
    const Real initial = [&] {
        ConstraintSolver solver(fb.model, fb.closure);
        solver.evaluate(data, q_bad, VecX::Zero(3), 0.0);
        return solver.phi().norm();
    }();
    CHECK(final_error(10.0, 10.0) < 1e-3 * initial);
    CHECK(final_error(0.0, 0.0) > 0.5 * initial);
}

TEST_CASE("Kernel constrained dynamics: no allocation after setup", "[.][kernel][alloc]")
{
    // Hidden: run by the CI job kernel-no-malloc, a Debug build with
    // EIGEN_RUNTIME_NO_MALLOC, in which any Eigen heap allocation while
    // set_is_malloc_allowed(false) fails an assertion.
#if !defined(EIGEN_RUNTIME_NO_MALLOC) || defined(NDEBUG)
    FAIL("needs a build with EIGEN_RUNTIME_NO_MALLOC and assertions on");
#else
#if defined(_MSC_VER) && defined(_DEBUG)
    // Report a failed assertion on stderr and exit, instead of in a dialog box.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    // Each call is named on stderr first, so a failed assertion shows which
    // one allocated.
    auto check = [](const char* what, auto&& call) {
        std::fprintf(stderr, "no allocation in %s?\n", what);
        Eigen::internal::set_is_malloc_allowed(false);
        call();
        Eigen::internal::set_is_malloc_allowed(true);
    };

    // A free root with every joint below it, and the four-bar with its
    // redundant closure.
    Rng rng(53);
    const Model tree = mbd_test::random_tree(rng, mbd_test::KJoint::Free, mbd_test::all_kernel_joints());
    Data td(tree);
    const VecX tq = mbd_test::random_configuration(tree, rng);
    const VecX tv = mbd_test::random_vector(tree.nv, rng, 1.0);
    const VecX ta = mbd_test::random_vector(tree.nv, rng, 1.0);
    VecX tqd(tree.nq), tq1 = tq, tdv(tree.nv);
    check("forward_kinematics", [&] { forward_kinematics(tree, td, tq, tv, ta); });
    check("rnea", [&] { rnea(tree, td, tq, tv, ta); });
    check("crba", [&] { crba(tree, td, tq); });
    check("aba", [&] { aba(tree, td, tq, tv, ta); });
    check("q_dot", [&] { q_dot(tree, tq, tv, tqd); });
    check("integrate", [&] { integrate(tree, tq, tv, 0.01, tq1); });
    check("difference", [&] { difference(tree, tq, tq1, tdv); });

    FourBar fb;
    Data data(fb.model);
    ConstraintSolver solver(fb.model, fb.closure);
    VecX q = fb.q, v = Eigen::Vector3d(2.0, 0.0, 0.0);
    const VecX tau = VecX::Zero(3);
    solver.project(data, q, v, 0.0);
    check("ConstraintSolver::evaluate", [&] { solver.evaluate(data, q, v, 0.0); });
    check("ConstraintSolver::forward_dynamics", [&] { solver.forward_dynamics(data, q, v, tau, 0.0); });
    check("ConstraintSolver::project", [&] { solver.project(data, q, v, 0.0); });
    SUCCEED("no Eigen allocation after setup");
#endif
}
