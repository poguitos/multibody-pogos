#pragma once

// The least change of coordinates or velocities that the constraints allow,
// with chosen velocities held, and Gauss-Newton projection onto the
// constraints built from it. Shared by the initial-condition assembly
// (assembly.cpp) and statics (statics.cpp). Internal to the library.

#include <cmath>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel::detail {

/// The weights of corrections: the diagonal of the mass matrix at q. Throws
/// MBD-K061, naming `function`, where an entry is not positive.
VecX mass_weights(const char* function, const Model& model, Data& data, const VecX& q);

/// The least changes x of the velocities not held, in the norm
/// sqrt(x^T W x) with W diagonal. With B = J_f W_f^-1/2 and A = B B^T,
/// x_f = W_f^-1/2 B^T mu with A mu = r solves J_f x_f = r: B^T mu lies in the
/// range of B^T, so it is the least solution, whichever mu solves A mu = r.
/// RankRevealingLDLT gives mu and the rank of J_f by the constraint solver's
/// own rule; when r is out of reach (held coordinates that over-determine the
/// constraints), it drops the dependent equations.
class LeastChange {
public:
    LeastChange(int nv, const std::vector<int>& held) : nv_(nv)
    {
        std::vector<bool> is_held(static_cast<std::size_t>(nv), false);
        for (int k : held) is_held[static_cast<std::size_t>(k)] = true;
        for (int k = 0; k < nv; ++k) {
            if (!is_held[static_cast<std::size_t>(k)]) free_.push_back(k);
        }
        scale_.resize(static_cast<Index>(free_.size()));
    }

    void factorize(const MatX& J, const VecX& w)
    {
        const Index nf = static_cast<Index>(free_.size());
        B_.resize(J.rows(), nf);
        for (Index c = 0; c < nf; ++c) {
            const int k = free_[static_cast<std::size_t>(c)];
            scale_(c) = 1.0 / std::sqrt(w(k));
            B_.col(c) = scale_(c) * J.col(k);
        }
        empty_ = B_.rows() == 0 || B_.cols() == 0;
        if (empty_) return;
        A_.noalias() = B_ * B_.transpose();
        ldlt_.compute(A_);
    }

    /// x (every velocity, sized here): the least change with J_f x_f = r;
    /// zero where held.
    void solve(const VecX& r, VecX& x)
    {
        x.setZero(nv_);
        if (empty_) return;
        ldlt_.solve(r, mu_);
        y_.noalias() = B_.transpose() * mu_;
        for (std::size_t c = 0; c < free_.size(); ++c) {
            x(free_[c]) = scale_(static_cast<Index>(c)) * y_(static_cast<Index>(c));
        }
    }

    int rank() const { return empty_ ? 0 : ldlt_.rank(); }

private:
    int nv_;
    std::vector<int> free_;
    VecX scale_, mu_, y_;
    MatX B_, A_;
    RankRevealingLDLT ldlt_;
    bool empty_{true};
};

/// Gauss-Newton onto phi(q, t) = 0, changing only what `change` leaves free:
/// each step the least that zeroes the linearized constraints in the weights
/// w, halved until |phi| decreases (as ConstraintSolver::project). Returns the
/// iterations taken; `residual` is |phi| at the end, and the solver holds phi
/// and J at the final q.
int project_positions(const Model& model, ConstraintSolver& solver, Data& data,
                      LeastChange& change, const VecX& w, VecX& q, Real t,
                      Real tolerance, int max_iterations, Real& residual);

} // namespace mbd::kernel::detail
