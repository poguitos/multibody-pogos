#pragma once

// The mass matrix of a tree, factorized without fill-in (plan task 4.8,
// decision D39): Featherstone's LTL factorization, "Rigid Body Dynamics
// Algorithms" (2008), section 6.5.
//
// A tree's joint-space mass matrix has entries M(i, j) only where velocity j
// is velocity i's, one of its ancestors' or one of its descendants' (its
// joint, a joint on its path to the root, or one below it): M is as sparse
// as the tree is branched. Number the velocities so that each comes after
// its parent (the model's order does), and M = L^T L with L lower
// triangular has the same pattern as M's lower triangle: no fill-in.
// Factorizing from the last velocity to the first touches only each
// velocity's ancestors: about sum_k d_k^2 / 2 multiply-adds, d_k the number
// of k's ancestors, against n^3 / 6 dense; the solves are sums along the
// same paths. A chain gains nothing (every velocity is every later one's
// ancestor); a car's four corners, each hung from the chassis, gain much.
//
// Velocity-level parents: a joint's first velocity has as parent the last
// velocity of the nearest joint above it that has any; each of its other
// velocities has the one before it (a joint's velocities form a chain).
//
// Storage: L^T, column k holding L's row k, so that the loops over a
// velocity's ancestors run down a column. A path to the root is a few runs
// of consecutively numbered velocities (a joint's, a chain of bodies
// numbered in order); each run is one contiguous stretch of a column, worked
// by a plain loop the compiler vectorizes (an Eigen segment operation per
// run cost more than the scalar work on a tree's short runs: D39).

#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel {

class TreeCholesky {
public:
    /// The velocity-level parents of `model`, and storage for its nv x nv
    /// matrices. Allocates; nothing else does.
    explicit TreeCholesky(const Model& model);

    /// Factorize M = L^T L, reading only M's lower triangle on the tree's
    /// pattern. false if M is not positive definite there.
    bool compute(const MatX& M);

    /// x := M^-1 x.
    void solve(VecX& x) const;
    /// X := L^-T X, column by column: with R = L^T, M = R R^T, the R^-1 of
    /// the constraint solver's Z = R^-1 J^T.
    template <class Matrix>
    void solve_Lt(Matrix& X) const;
    /// x := L^-1 x: the R^-T that maps Z mu back to M^-1 J^T mu.
    void solve_L(VecX& x) const;

    /// L^T, in the upper triangle: zero off the tree's pattern.
    const MatX& Lt() const { return U_; }
    /// Velocity k's parent, -1 at a root.
    Index parent(Index k) const { return parent_[static_cast<std::size_t>(k)]; }
    /// The first velocity of the run of consecutively numbered ancestors
    /// that ends at k: k's ancestors from start(k) to k - 1 are contiguous,
    /// and the next one is parent(start(k)).
    Index start(Index k) const { return start_[static_cast<std::size_t>(k)]; }
    /// The factorization's multiply-adds, and the dense Cholesky's (n^3 / 6),
    /// for choosing between them.
    Real work() const { return work_; }
    Real dense_work() const { return dense_work_; }

private:
    std::vector<Index> parent_, start_;
    MatX U_;   ///< L^T in the upper triangle, on the tree's pattern
    Real work_{0.0}, dense_work_{0.0};
};

template <class Matrix>
void TreeCholesky::solve_Lt(Matrix& X) const
{
    // L^T y = b: L^T's column k holds L(k, i) at its ancestors i, so from
    // the last velocity up: y_k = b_k / L(k, k), then b_i -= L(k, i) y_k,
    // run by run.
    const Index n = U_.rows();
    const Real* U = U_.data();
    for (Index c = 0; c < X.cols(); ++c) {
        Real* xc = &X(0, c);
        for (Index k = n - 1; k >= 0; --k) {
            if (xc[k] == 0.0) continue;   // a sparse right-hand side (a constraint's row of J)
            const Real* uk = U + k * n;
            const Real y = xc[k] / uk[k];
            xc[k] = y;
            for (Index e = parent(k); e >= 0; e = parent(start(e))) {
                for (Index j = start(e); j <= e; ++j) xc[j] -= y * uk[j];
            }
        }
    }
}

} // namespace mbd::kernel
