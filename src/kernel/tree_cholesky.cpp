#include "mbd/kernel/tree_cholesky.hpp"

#include <cmath>

namespace mbd::kernel {

TreeCholesky::TreeCholesky(const Model& model)
{
    const Index n = model.nv;
    parent_.assign(static_cast<std::size_t>(n), -1);
    // last[i]: the last velocity of body i's joint or of the nearest joint
    // above it that has any; -1 if none.
    std::vector<Index> last(static_cast<std::size_t>(model.nbodies()), -1);
    for (int i = 1; i < model.nbodies(); ++i) {
        Index above = last[static_cast<std::size_t>(model.parent[i])];
        for (int s = 0; s < model.nvs[i]; ++s) {
            const Index k = model.idx_v[i] + s;
            parent_[static_cast<std::size_t>(k)] = above;
            above = k;
        }
        last[static_cast<std::size_t>(i)] = above;
    }
    // A run continues down to k while each velocity's parent is the one
    // numbered just before it.
    start_.assign(static_cast<std::size_t>(n), 0);
    for (Index k = 0; k < n; ++k) {
        start_[static_cast<std::size_t>(k)] = k > 0 && parent(k) == k - 1 ? start(k - 1) : k;
    }
    U_ = MatX::Zero(n, n);
    // The work: velocity k's elimination updates the d (d + 1) / 2 entries
    // among its d ancestors.
    for (Index k = 0; k < n; ++k) {
        Real d = 0.0;
        for (Index i = parent(k); i >= 0; i = parent(i)) d += 1.0;
        work_ += 0.5 * d * (d + 1.0) + d + 1.0;
    }
    dense_work_ = static_cast<Real>(n) * static_cast<Real>(n) * static_cast<Real>(n) / 6.0;
}

bool TreeCholesky::compute(const MatX& M)
{
    const Index n = U_.rows();
    Real* U = U_.data();
    // M's lower triangle on the pattern, transposed: column k of U holds row
    // k of M at k and its ancestors.
    for (Index k = 0; k < n; ++k) {
        Real* uk = U + k * n;
        uk[k] = M(k, k);
        for (Index i = parent(k); i >= 0; i = parent(i)) uk[i] = M(k, i);
    }
    // RBDA table 6.3, from the last velocity to the first, on L^T: k's
    // column scaled, then each ancestor i's column, at i and its ancestors,
    // less L(k, i) times k's column there. Within a run i's ancestors are
    // the run's start to i, contiguous; the runs above are all of them.
    // Plain loops over each run: short runs (a tree's) cost no more than
    // the scalar walk, long ones (a chain's) are vectorized.
    for (Index k = n - 1; k >= 0; --k) {
        Real* uk = U + k * n;
        if (!(uk[k] > 0.0)) return false;
        const Real lkk = std::sqrt(uk[k]);
        uk[k] = lkk;
        for (Index i = parent(k); i >= 0; i = parent(i)) uk[i] /= lkk;
        for (Index e = parent(k); e >= 0; e = parent(start(e))) {
            const Index b = start(e);
            for (Index i = e; i >= b; --i) {
                Real* ui = U + i * n;
                const Real lki = uk[i];
                for (Index j = b; j <= i; ++j) ui[j] -= lki * uk[j];
                for (Index f = parent(b); f >= 0; f = parent(start(f))) {
                    for (Index j = start(f); j <= f; ++j) ui[j] -= lki * uk[j];
                }
            }
        }
    }
    return true;
}

void TreeCholesky::solve_L(VecX& x) const
{
    // L x = y, forward: x_k = (y_k - sum over k's ancestors i of L(k, i)
    // x_i) / L(k, k), the sum run by run down column k of L^T.
    const Index n = U_.rows();
    const Real* U = U_.data();
    Real* xp = x.data();
    for (Index k = 0; k < n; ++k) {
        const Real* uk = U + k * n;
        Real s = xp[k];
        for (Index e = parent(k); e >= 0; e = parent(start(e))) {
            for (Index j = start(e); j <= e; ++j) s -= uk[j] * xp[j];
        }
        xp[k] = s / uk[k];
    }
}

void TreeCholesky::solve(VecX& x) const
{
    // M = L^T L: L^-T, then L^-1.
    solve_Lt(x);
    solve_L(x);
}

} // namespace mbd::kernel
