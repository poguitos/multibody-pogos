// The mass matrix factorized along the tree (plan task 4.8, decision D39):
// Featherstone's M = L^T L, which keeps the zeros the tree's branches leave
// in M.
//
//   - on every joint type, in chains and branching trees, L has exactly the
//     tree's pattern (zero wherever a velocity is not the other's ancestor),
//     the runs of consecutively numbered ancestors list each velocity's
//     ancestors, L^T L is M to rounding, and the solves equal the dense
//     Cholesky's;
//   - the constraint solver gives the same accelerations, multipliers and
//     projections with either factorization, on a closed four-bar and on
//     trees;
//   - the work estimate that chooses between them: a chain gains nothing, a
//     branching tree does; and Automatic's choice on either side of its
//     two thresholds.

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Cholesky>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/tree_cholesky.hpp"

#include "kernel/four_bar.hpp"
#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::KJoint;
using mbd_test::Rng;
using mbd_test::all_kernel_joints;
using mbd_test::random_chain;
using mbd_test::random_configuration;
using mbd_test::random_tree;
using mbd_test::random_vector;

namespace {

/// Every joint type as the root of a branching tree and as a chain's middle,
/// and a binary tree of 31 revolute bodies (the benchmark's shape).
std::vector<Model> sparse_models(Rng& rng)
{
    std::vector<Model> models;
    for (KJoint root : all_kernel_joints()) {
        models.push_back(random_tree(rng, root, all_kernel_joints()));
        models.push_back(random_chain(rng, KJoint::Revolute, {root, KJoint::Spherical, KJoint::Revolute}));
    }
    Model binary;
    for (int i = 1; i <= 31; ++i) mbd_test::add_random_body(binary, rng, KJoint::Revolute, i / 2);
    models.push_back(binary);
    return models;
}

/// Whether velocity i is velocity k's ancestor or k itself.
bool on_path(const TreeCholesky& tc, Index k, Index i)
{
    for (Index j = k; j >= 0; j = tc.parent(j)) {
        if (j == i) return true;
    }
    return false;
}

} // namespace

TEST_CASE("Tree Cholesky: L has the tree's pattern, L^T L is M, and the solves are the dense ones",
          "[kernel][sparse]")
{
    Rng rng(21);
    for (const Model& model : sparse_models(rng)) {
        Data data(model);
        const VecX q = random_configuration(model, rng);
        const MatX M = crba(model, data, q);
        TreeCholesky tc(model);
        REQUIRE(tc.compute(M));
        const MatX L = tc.Lt().transpose();
        INFO(model.nbodies() - 1 << " bodies, " << model.nv << " velocities; work " << tc.work() << " against "
                                  << tc.dense_work() << " dense");
        // The runs list each velocity's ancestors exactly: contiguous from
        // start(e) to e, then the next run from parent(start(e)).
        int run_mismatches = 0;
        for (Index k = 0; k < model.nv; ++k) {
            std::vector<Index> by_parent, by_run;
            for (Index i = tc.parent(k); i >= 0; i = tc.parent(i)) by_parent.push_back(i);
            for (Index e = tc.parent(k); e >= 0; e = tc.parent(tc.start(e))) {
                for (Index i = e; i >= tc.start(e); --i) by_run.push_back(i);
            }
            if (by_run != by_parent) ++run_mismatches;
        }
        CHECK(run_mismatches == 0);
        int outside = 0;
        for (Index k = 0; k < model.nv; ++k) {
            for (Index i = 0; i < model.nv; ++i) {
                if (!(i <= k && on_path(tc, k, i)) && L(k, i) != 0.0) ++outside;
            }
        }
        CHECK(outside == 0);
        CHECK((L.transpose() * L - M).norm() < 1e-13 * M.norm());

        const VecX b = random_vector(model.nv, rng, 1.0);
        VecX x = b;
        tc.solve(x);
        const VecX dense = M.llt().solve(b);
        CHECK((x - dense).norm() < 1e-12 * (1.0 + dense.norm()));
        // L^-1 L^-T b is the same solve in two halves.
        VecX y = b;
        tc.solve_Lt(y);
        tc.solve_L(y);
        CHECK((y - dense).norm() < 1e-12 * (1.0 + dense.norm()));
    }
}

TEST_CASE("Tree Cholesky: a chain gains nothing, a branching tree gains its branches", "[kernel][sparse]")
{
    // The work estimate counts each velocity's ancestors: in a chain of n
    // revolute joints velocity k has k - 1 of them, sum (d^2 + d) / 2 + d + 1
    // = n (n + 1) (n + 2) / 6, about the dense n^3 / 6; in the binary tree of
    // 31 bodies the deepest velocity has 4.
    Rng rng(22);
    Model chain;
    for (int i = 1; i <= 16; ++i) mbd_test::add_random_body(chain, rng, KJoint::Revolute, i - 1);
    const TreeCholesky tc_chain(chain);
    const Real n = 16.0;
    CHECK(tc_chain.work() == n * (n + 1.0) * (n + 2.0) / 6.0);
    CHECK(tc_chain.dense_work() == n * n * n / 6.0);

    Model binary;
    for (int i = 1; i <= 31; ++i) mbd_test::add_random_body(binary, rng, KJoint::Revolute, i / 2);
    const TreeCholesky tc_tree(binary);
    INFO("binary tree of 31: work " << tc_tree.work() << " against " << tc_tree.dense_work());
    CHECK(tc_tree.work() < 0.1 * tc_tree.dense_work());
}

TEST_CASE("Constraint solver: the tree's factorization gives the dense one's dynamics and projections",
          "[kernel][sparse][constrained]")
{
    // The four-bar with a pendulum, closed in 3D (5 equations of rank 2), at
    // a state off its constraints, and the trees: forward dynamics and the
    // projection with each factorization, compared.
    Rng rng(23);
    std::vector<Model> models = sparse_models(rng);
    mbd_test::FourBar fb(true);
    for (std::size_t m = 0; m <= models.size(); ++m) {
        const bool four_bar = m == models.size();
        const Model& model = four_bar ? fb.sys.model : models[m];
        const auto constraints =
            four_bar ? fb.sys.constraints : std::vector<std::shared_ptr<const ConstraintModel>>{};
        VecX q = four_bar ? VecX(model.neutral_configuration()) : random_configuration(model, rng);
        if (four_bar) {
            q.head(3) = mbd_test::FourBar::configuration(1.0) + Eigen::Vector3d(0.01, -0.02, 0.015);
            q(3) = 0.3;
        }
        const VecX v = random_vector(model.nv, rng, 1.5);
        const VecX tau = random_vector(model.nv, rng, 2.0);
        Data data_dense(model), data_tree(model);
        ConstraintSolver dense(model, constraints), tree(model, constraints);
        dense.mass_factorization = ConstraintSolver::MassFactorization::Dense;
        tree.mass_factorization = ConstraintSolver::MassFactorization::Tree;
        const VecX a_dense = dense.forward_dynamics(data_dense, q, v, tau, 0.0);
        const VecX a_tree = tree.forward_dynamics(data_tree, q, v, tau, 0.0);
        INFO((four_bar ? "four-bar" : "tree ") << m << ": |a| " << a_dense.norm() << ", difference "
                                               << (a_tree - a_dense).norm());
        CHECK(tree.tree_factorization());
        CHECK(!dense.tree_factorization());
        CHECK((a_tree - a_dense).norm() < 1e-11 * (1.0 + a_dense.norm()));
        if (!four_bar) continue;
        CHECK((tree.lambda() - dense.lambda()).norm() < 1e-11 * (1.0 + dense.lambda().norm()));
        VecX q_dense = q, v_dense = v, q_tree = q, v_tree = v;
        const ProjectionInfo p_dense = dense.project(data_dense, q_dense, v_dense, 0.0);
        const ProjectionInfo p_tree = tree.project(data_tree, q_tree, v_tree, 0.0);
        CHECK(p_dense.converged);
        CHECK(p_tree.converged);
        CHECK(p_tree.iterations == p_dense.iterations);
        CHECK((q_tree - q_dense).norm() < 1e-12);
        CHECK((v_tree - v_dense).norm() < 1e-11 * (1.0 + v_dense.norm()));
    }
}

TEST_CASE("Constraint solver: Automatic takes the tree's factorization except on a long chain",
          "[kernel][sparse][constrained]")
{
    // The rule: the tree's unless the model has more than 40 velocities and
    // the tree's work is at least 0.8 of the dense one's. A chain of n has
    // the ratio (n + 1) (n + 2) / n^2: 1.08 at 40, still the tree's by its
    // size; 1.07 at 41, dense. A binary tree of 63 and a root carrying two
    // chains of 32 (65 velocities) take the tree's. Their work, each velocity
    // d deep costing d (d + 1) / 2 + d + 1: a chain n (n + 1) (n + 2) / 6; the
    // binary tree sum over its levels L of 2^L (L (L + 1) / 2 + L + 1) = 1023
    // (0.025 of 63^3 / 6); the fork 1 + 2 sum_{d=1}^{32} (d (d + 1) / 2 + d + 1)
    // = 13089 (0.29 of 65^3 / 6).
    Rng rng(24);
    struct Case {
        std::string name;
        Model model;
        Real work;
        bool tree;
    };
    std::vector<Case> cases;
    for (int n : {40, 41}) {
        Model chain;
        for (int i = 1; i <= n; ++i) mbd_test::add_random_body(chain, rng, KJoint::Revolute, i - 1);
        cases.push_back({"chain of " + std::to_string(n), chain, n * (n + 1.0) * (n + 2.0) / 6.0, n == 40});
    }
    Model binary;
    for (int i = 1; i <= 63; ++i) mbd_test::add_random_body(binary, rng, KJoint::Revolute, i / 2);
    cases.push_back({"binary tree of 63", binary, 1023.0, true});
    Model fork;
    for (int i = 1; i <= 65; ++i) {
        const int parent = i == 1 ? 0 : i == 34 ? 1 : i - 1;   // the second chain from the root too
        mbd_test::add_random_body(fork, rng, KJoint::Revolute, parent);
    }
    cases.push_back({"root and two chains of 32", fork, 13089.0, true});

    for (const Case& c : cases) {
        ConstraintSolver solver(c.model, {});
        Data data(c.model);
        const VecX q = random_configuration(c.model, rng);
        const VecX v = random_vector(c.model.nv, rng, 1.0);
        const VecX tau = random_vector(c.model.nv, rng, 1.0);
        solver.forward_dynamics(data, q, v, tau, 0.0);
        const Real ratio = solver.tree_cholesky().work() / solver.tree_cholesky().dense_work();
        INFO(c.name << ": " << c.model.nv << " velocities, work ratio " << ratio);
        CHECK(solver.tree_cholesky().work() == c.work);
        const Real n = static_cast<Real>(c.model.nv);
        CHECK(solver.tree_cholesky().dense_work() == n * n * n / 6.0);
        CHECK(solver.tree_factorization() == c.tree);
    }
}
