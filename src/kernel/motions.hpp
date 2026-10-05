#pragma once

// The motions a constrained system allows at a configuration: an orthonormal
// basis N of the velocities with J N = 0 and no component along held
// coordinates. Statics (task 3.5) and linearisation (task 3.7) work in the
// coordinates it defines.

#include <vector>

#include <Eigen/SVD>

#include "mbd/core/core.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

namespace mbd::kernel::motions {

/// Singular values of the constraint and hold rows below this fraction of
/// the largest belong to dependent rows.
inline constexpr Real kRowRankTolerance = 1e-8;

/// N (nv x d) at (q, t), from an SVD of the constraint rows and one row per
/// held coordinate. `data` is used as working storage.
inline MatX allowed(ConstraintSolver& solver, Data& data, const VecX& q, Real t,
                    const std::vector<int>& held, int nv)
{
    const int m = solver.size();
    const int h = static_cast<int>(held.size());
    if (m + h == 0) return MatX::Identity(nv, nv);
    MatX rows = MatX::Zero(m + h, nv);
    if (m > 0) {
        solver.evaluate(data, q, VecX::Zero(nv), t);
        rows.topRows(m) = solver.J();
    }
    for (int i = 0; i < h; ++i) rows(m + i, held[static_cast<std::size_t>(i)]) = 1.0;
    const Eigen::JacobiSVD<MatX> svd(rows, Eigen::ComputeFullV);
    const VecX& s = svd.singularValues();
    int rank = 0;
    for (Index k = 0; k < s.size(); ++k) {
        if (s(k) > kRowRankTolerance * s(0)) ++rank;
    }
    return svd.matrixV().rightCols(nv - rank);
}

} // namespace mbd::kernel::motions
