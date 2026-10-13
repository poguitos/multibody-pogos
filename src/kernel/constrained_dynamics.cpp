#include "mbd/kernel/constrained_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "mbd/kernel/algorithms.hpp"

#include "checks.hpp"

namespace mbd::kernel {

namespace {

// The number of equations. Every constraint must exist and act on bodies of
// the model: a marker on a missing body would be read out of bounds.
int total_size(const Model& model,
               const std::vector<std::shared_ptr<const ConstraintModel>>& constraints)
{
    int m = 0;
    for (std::size_t k = 0; k < constraints.size(); ++k) {
        const auto& c = constraints[k];
        MBD_THROW_IF(!c, "MBD-K030: kernel::ConstraintSolver: constraint " + std::to_string(k) + " is empty");
        for (int b : c->bodies()) {
            MBD_THROW_IF(b < 0 || b >= model.nbodies(),
                         "MBD-K031: kernel::ConstraintSolver: constraint " + std::to_string(k) + " ("
                             + c->name() + ") refers to body " + std::to_string(b)
                             + ", but the model's bodies are 0 to "
                             + std::to_string(model.nbodies() - 1) + ".");
        }
        m += c->size();
    }
    return m;
}

} // namespace

RankRevealingLDLT::RankRevealingLDLT(Index n)
    : LD_(MatX::Zero(n, n))
    , pivot_(static_cast<std::size_t>(n))
    , work_(VecX::Zero(n))
{
}

void RankRevealingLDLT::compute(const MatX& A)
{
    const Index n = A.rows();
    if (LD_.rows() != n) {
        pivot_.resize(static_cast<std::size_t>(n));
        work_.resize(n);
    }
    // In the lower triangle only: L goes below the diagonal as it is found,
    // D on it, and the trailing block holds the Schur complement.
    LD_ = A;
    for (Index i = 0; i < n; ++i) pivot_[static_cast<std::size_t>(i)] = i;
    rank_ = 0;
    Real first = 0.0;
    for (Index k = 0; k < n; ++k) {
        Index j = k;
        for (Index i = k + 1; i < n; ++i) {
            if (LD_(i, i) > LD_(j, j)) j = i;
        }
        const Real d = LD_(j, j);
        if (k == 0) first = d;
        // Below this, what is left is rounding error: stop.
        if (!(first > 0.0) || !(d > kRankTolerance * first)) break;
        if (j != k) {
            // Swap rows and columns k and j of the symmetric matrix, as
            // stored in its lower triangle (LAPACK's syswapr).
            for (Index c = 0; c < k; ++c) std::swap(LD_(k, c), LD_(j, c));
            std::swap(LD_(k, k), LD_(j, j));
            for (Index i = k + 1; i < j; ++i) std::swap(LD_(i, k), LD_(j, i));
            for (Index i = j + 1; i < n; ++i) std::swap(LD_(i, k), LD_(i, j));
            std::swap(pivot_[static_cast<std::size_t>(k)], pivot_[static_cast<std::size_t>(j)]);
        }
        for (Index i = k + 1; i < n; ++i) LD_(i, k) /= d;
        for (Index c = k + 1; c < n; ++c) {
            const Real s = LD_(c, k) * d;
            for (Index i = c; i < n; ++i) LD_(i, c) -= LD_(i, k) * s;
        }
        rank_ = static_cast<int>(k + 1);
    }
}

void RankRevealingLDLT::solve(const VecX& r, VecX& x) const
{
    // x = P L^-T D^+ L^-1 P^T r, D^+ zero past the rank. When r is in the
    // range of A this solves A x = r exactly.
    const Index n = LD_.rows();
    const Index rank = rank_;
    for (Index i = 0; i < n; ++i) work_(i) = r(pivot_[static_cast<std::size_t>(i)]);
    for (Index k = 0; k < rank; ++k) {
        const Real wk = work_(k);
        for (Index i = k + 1; i < n; ++i) work_(i) -= LD_(i, k) * wk;
    }
    for (Index k = 0; k < n; ++k) work_(k) = k < rank ? work_(k) / LD_(k, k) : 0.0;
    for (Index k = rank - 1; k >= 0; --k) {
        Real s = work_(k);
        for (Index i = k + 1; i < rank; ++i) s -= LD_(i, k) * work_(i);
        work_(k) = s;
    }
    x.resize(n);
    for (Index i = 0; i < n; ++i) x(pivot_[static_cast<std::size_t>(i)]) = work_(i);
}

ConstraintSolver::ConstraintSolver(const Model& model,
                                   std::vector<std::shared_ptr<const ConstraintModel>> constraints)
    : model_(model)
    , constraints_(std::move(constraints))
    , m_(total_size(model, constraints_))
    , llt_M_(model.nv)
    , ldlt_A_(m_)
{
    const int nv = model.nv;
    phi_.setZero(m_);
    nu_.setZero(m_);
    gamma_.setZero(m_);
    lambda_.setZero(m_);
    rhs_m_.setZero(m_);
    mu_.setZero(m_);
    J_.setZero(m_, nv);
    Z_.setZero(nv, m_);
    A_.setZero(m_, m_);
    q_save_.setZero(model.nq);
    v_dot_.setZero(nv);
    v_dot_free_.setZero(nv);
    rhs_v_.setZero(nv);
    dv_.setZero(nv);
    zero_v_.setZero(nv);
    work_v_.setZero(nv);
    info_.equations = m_;
}

void ConstraintSolver::evaluate(Data& data, const VecX& q, const VecX& v, Real t)
{
    checks::data("kernel::ConstraintSolver::evaluate", model_, data);
    checks::q("kernel::ConstraintSolver::evaluate", model_, q);
    checks::v("kernel::ConstraintSolver::evaluate", model_, v);
    // Zero joint accelerations: the body accelerations are then the
    // velocity-product terms the constraints need for gamma.
    forward_kinematics(model_, data, q, v, zero_v_);
    calc_constraints(data, t);
}

void ConstraintSolver::calc_constraints(const Data& data, Real t)
{
    Index r = 0;
    for (const auto& c : constraints_) {
        const Index m = c->size();
        c->calc(model_, data, t, phi_.segment(r, m), J_.middleRows(r, m),
                nu_.segment(r, m), gamma_.segment(r, m));
        r += m;
    }
}

void ConstraintSolver::add_constraint_motion(const VecX& x, VecX& out)
{
    work_v_.noalias() = Z_ * x;
    llt_M_.matrixU().solveInPlace(work_v_);
    out += work_v_;
}

void ConstraintSolver::factorize_constraint_matrix()
{
    // With M = L L^T, J M^-1 J^T = (L^-1 J^T)^T (L^-1 J^T): one triangular
    // solve, and M^-1 J^T is never needed as a matrix.
    Z_ = J_.transpose();
    llt_M_.matrixL().solveInPlace(Z_);
    A_.noalias() = Z_.transpose() * Z_;
    // The equations behind dropped pivots are redundant.
    ldlt_A_.compute(A_);
    info_.rank = ldlt_A_.rank();
}

void ConstraintSolver::solve_constraint_matrix(const VecX& r, VecX& x)
{
    ldlt_A_.solve(r, x);
}

const VecX& ConstraintSolver::forward_dynamics(Data& data, const VecX& q, const VecX& v,
                                               const VecX& tau, Real t)
{
    checks::q("kernel::ConstraintSolver::forward_dynamics", model_, q);
    checks::v("kernel::ConstraintSolver::forward_dynamics", model_, v);
    checks::v("kernel::ConstraintSolver::forward_dynamics", model_, tau, "tau");
    checks::data("kernel::ConstraintSolver::forward_dynamics", model_, data);
    // Zero joint accelerations: the body accelerations are then the
    // velocity-product terms the constraints need for gamma.
    forward_kinematics(model_, data, q, v, zero_v_);
    return forward_dynamics_from_kinematics(data, v, tau, t);
}

const VecX& ConstraintSolver::forward_dynamics_from_kinematics(Data& data, const VecX& v,
                                                               const VecX& tau, Real t)
{
    checks::data("kernel::ConstraintSolver::forward_dynamics_from_kinematics", model_, data);
    checks::v("kernel::ConstraintSolver::forward_dynamics_from_kinematics", model_, v);
    checks::v("kernel::ConstraintSolver::forward_dynamics_from_kinematics", model_, tau, "tau");
    calc_constraints(data, t);

    // Unconstrained accelerations, from the one kinematics pass.
    mass_matrix(model_, data);
    bias_forces(model_, data);
    llt_M_.compute(data.M);
    mass_matrix_factorized_ = true;
    rhs_v_ = tau - data.tau;
    v_dot_free_ = rhs_v_;
    llt_M_.solveInPlace(v_dot_free_);
    if (m_ == 0) {
        v_dot_ = v_dot_free_;
        return v_dot_;
    }

    // (J M^-1 J^T) lambda = gamma_stabilized - J v_dot_free.
    factorize_constraint_matrix();
    if (quasi_static) rhs_m_.setZero();
    else rhs_m_ = gamma_;
    if (baumgarte_alpha != 0.0) {
        rhs_m_.noalias() -= (2.0 * baumgarte_alpha) * (J_ * v);
        rhs_m_ += (2.0 * baumgarte_alpha) * nu_;
    }
    if (baumgarte_beta != 0.0) rhs_m_ -= (baumgarte_beta * baumgarte_beta) * phi_;
    rhs_m_.noalias() -= J_ * v_dot_free_;
    solve_constraint_matrix(rhs_m_, lambda_);

    v_dot_ = v_dot_free_;
    add_constraint_motion(lambda_, v_dot_);
    return v_dot_;
}

ProjectionInfo ConstraintSolver::project(Data& data, VecX& q, VecX& v, Real t,
                                         Real tolerance, int max_iterations)
{
    checks::data("kernel::ConstraintSolver::project", model_, data);
    checks::q("kernel::ConstraintSolver::project", model_, q);
    checks::v("kernel::ConstraintSolver::project", model_, v);
    ProjectionInfo out;
    if (m_ == 0) {
        out.converged = true;
        return out;
    }

    // One mass matrix for the whole projection: that of q, or the one the
    // last forward_dynamics factorized (see reuse_mass_matrix).
    if (!(reuse_mass_matrix && mass_matrix_factorized_)) {
        crba(model_, data, q);
        llt_M_.compute(data.M);
        mass_matrix_factorized_ = true;
    }

    // Positions: Gauss-Newton steps dq = -M^-1 J^T (J M^-1 J^T)^-1 phi, each
    // halved until |phi| decreases, for starts far from the solution.
    evaluate(data, q, zero_v_, t);
    out.position_residual = phi_.norm();
    out.initial_position_residual = out.position_residual;
    while (out.position_residual > tolerance && out.iterations < max_iterations) {
        factorize_constraint_matrix();
        solve_constraint_matrix(phi_, mu_);
        dv_.setZero();
        add_constraint_motion(mu_, dv_);
        q_save_ = q;
        const Real before = out.position_residual;
        Real step = 1.0;
        for (int halving = 0; halving < 12; ++halving) {
            integrate(model_, q_save_, dv_, -step, q);
            evaluate(data, q, zero_v_, t);
            out.position_residual = phi_.norm();
            if (out.position_residual < before) break;
            step *= 0.5;
        }
        ++out.iterations;
    }
    out.converged = out.position_residual <= tolerance;

    // Velocities: v += M^-1 J^T (J M^-1 J^T)^-1 (nu - J v). J and nu at the
    // final q are those of its last evaluation: nu does not depend on v.
    factorize_constraint_matrix();
    rhs_m_ = nu_;
    rhs_m_.noalias() -= J_ * v;
    solve_constraint_matrix(rhs_m_, mu_);
    add_constraint_motion(mu_, v);
    rhs_m_ = nu_;
    rhs_m_.noalias() -= J_ * v;
    out.velocity_residual = rhs_m_.norm();
    return out;
}

} // namespace mbd::kernel
