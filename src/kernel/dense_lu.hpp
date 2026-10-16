#pragma once

// A small dense LU for the integrators' iteration matrices (generalised-alpha,
// SDIRK4, Radau IIA, linearly implicit Euler), real or complex. Internal to
// the library.

#include <utility>
#include <vector>

#include "mbd/core/core.hpp"

namespace mbd::kernel::detail {

/// LU factorization of the square A with partial pivoting, in place (unit
/// lower and upper triangles; whole rows swapped, as LAPACK's getrf does),
/// real or complex. Returns false at a zero pivot. Allocates nothing.
template <class Matrix>
bool lu_factor(Matrix& A, std::vector<Index>& pivot)
{
    const Index n = A.rows();
    for (Index k = 0; k < n; ++k) {
        Index p = 0;
        const Real largest = A.col(k).tail(n - k).cwiseAbs().maxCoeff(&p);
        p += k;
        pivot[static_cast<std::size_t>(k)] = p;
        if (!(largest > 0.0)) return false;
        if (p != k) A.row(k).swap(A.row(p));
        const Index rest = n - k - 1;
        if (rest == 0) continue;
        A.col(k).tail(rest) /= A(k, k);
        A.bottomRightCorner(rest, rest).noalias() -= A.col(k).tail(rest) * A.row(k).tail(rest);
    }
    return true;
}

/// b := A^-1 b from lu_factor's result.
template <class Matrix, class Vector>
void lu_solve(const Matrix& LU, const std::vector<Index>& pivot, Vector& b)
{
    for (Index k = 0; k < LU.rows(); ++k) {
        const Index p = pivot[static_cast<std::size_t>(k)];
        if (p != k) std::swap(b(k), b(p));
    }
    LU.template triangularView<Eigen::UnitLower>().solveInPlace(b);
    LU.template triangularView<Eigen::Upper>().solveInPlace(b);
}

} // namespace mbd::kernel::detail
