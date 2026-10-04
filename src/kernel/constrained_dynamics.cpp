#include "mbd/kernel/constrained_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "mbd/kernel/algorithms.hpp"

namespace mbd::kernel {

namespace {

int total_size(const std::vector<std::shared_ptr<const ConstraintModel>>& constraints)
{
    int m = 0;
    for (const auto& c : constraints) {
        MBD_THROW_IF(!c, "kernel::ConstraintSolver: no constraint");
        m += c->size();
    }
    return m;
}

} // namespace

ConstraintSolver::ConstraintSolver(const Model& model,
                                   std::vector<std::shared_ptr<const ConstraintModel>> constraints)
    : model_(model)
    , constraints_(std::move(constraints))
    , m_(total_size(constraints_))
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
    Y_.setZero(nv, m_);
    A_.setZero(m_, m_);
    v_dot_.setZero(nv);
    v_dot_free_.setZero(nv);
    rhs_v_.setZero(nv);
    dv_.setZero(nv);
    zero_v_.setZero(nv);
    info_.equations = m_;
}

void ConstraintSolver::evaluate(Data& data, const VecX& q, const VecX& v, Real t)
{
    // Zero joint accelerations: the body accelerations are then the
    // velocity-product terms the constraints need for gamma.
    forward_kinematics(model_, data, q, v, zero_v_);
    Index r = 0;
    for (const auto& c : constraints_) {
        const Index m = c->size();
        c->calc(model_, data, t, phi_.segment(r, m), J_.middleRows(r, m),
                nu_.segment(r, m), gamma_.segment(r, m));
        r += m;
    }
}

void ConstraintSolver::factorize_constraint_matrix()
{
    Y_ = J_.transpose();
    llt_M_.solveInPlace(Y_);
    A_.noalias() = J_ * Y_;
    ldlt_A_.compute(A_);

    // Diagonal pivoting puts the largest pivots first; the equations behind
    // pivots that are negligible next to the largest are redundant.
    const auto D = ldlt_A_.vectorD();
    Real largest = 0.0;
    for (Index k = 0; k < m_; ++k) largest = std::max(largest, std::abs(D(k)));
    pivot_cut_ = kRankTolerance * largest;
    info_.rank = 0;
    for (Index k = 0; k < m_; ++k) {
        if (std::abs(D(k)) > pivot_cut_) ++info_.rank;
    }
}

void ConstraintSolver::solve_constraint_matrix(const VecX& r, VecX& x)
{
    // x = P^T L^-T D^+ L^-1 P r, with D^+ zero for the dropped pivots. When
    // r is in the range of A this solves A x = r exactly.
    x = ldlt_A_.transpositionsP() * r;
    ldlt_A_.matrixL().solveInPlace(x);
    const auto D = ldlt_A_.vectorD();
    for (Index k = 0; k < m_; ++k) {
        x(k) = std::abs(D(k)) > pivot_cut_ ? x(k) / D(k) : 0.0;
    }
    ldlt_A_.matrixU().solveInPlace(x);
    x = ldlt_A_.transpositionsP().transpose() * x;   // swaps in place
}

const VecX& ConstraintSolver::forward_dynamics(Data& data, const VecX& q, const VecX& v,
                                               const VecX& tau, Real t)
{
    MBD_ASSERT(q.size() == model_.nq && v.size() == model_.nv && tau.size() == model_.nv);
    evaluate(data, q, v, t);

    // Unconstrained accelerations.
    crba(model_, data, q);
    rnea(model_, data, q, v, zero_v_);
    llt_M_.compute(data.M);
    rhs_v_ = tau - data.tau;
    v_dot_free_ = rhs_v_;
    llt_M_.solveInPlace(v_dot_free_);
    if (m_ == 0) {
        v_dot_ = v_dot_free_;
        return v_dot_;
    }

    // (J M^-1 J^T) lambda = gamma_stabilized - J v_dot_free.
    factorize_constraint_matrix();
    rhs_m_ = gamma_;
    if (baumgarte_alpha != 0.0) {
        rhs_m_.noalias() -= (2.0 * baumgarte_alpha) * (J_ * v);
        rhs_m_ += (2.0 * baumgarte_alpha) * nu_;
    }
    if (baumgarte_beta != 0.0) rhs_m_ -= (baumgarte_beta * baumgarte_beta) * phi_;
    rhs_m_.noalias() -= J_ * v_dot_free_;
    solve_constraint_matrix(rhs_m_, lambda_);

    v_dot_ = v_dot_free_;
    v_dot_.noalias() += Y_ * lambda_;
    return v_dot_;
}

ProjectionInfo ConstraintSolver::project(Data& data, VecX& q, VecX& v, Real t,
                                         Real tolerance, int max_iterations)
{
    ProjectionInfo out;
    if (m_ == 0) {
        out.converged = true;
        return out;
    }

    // One mass matrix for the whole projection.
    crba(model_, data, q);
    llt_M_.compute(data.M);

    // Positions: Gauss-Newton steps dq = -M^-1 J^T (J M^-1 J^T)^-1 phi.
    for (;;) {
        evaluate(data, q, zero_v_, t);
        out.position_residual = phi_.norm();
        if (out.position_residual <= tolerance || out.iterations == max_iterations) break;
        factorize_constraint_matrix();
        solve_constraint_matrix(phi_, mu_);
        dv_.noalias() = Y_ * mu_;
        integrate(model_, q, dv_, -1.0, q);
        ++out.iterations;
    }
    out.converged = out.position_residual <= tolerance;

    // Velocities: v += M^-1 J^T (J M^-1 J^T)^-1 (nu - J v).
    evaluate(data, q, v, t);
    factorize_constraint_matrix();
    rhs_m_ = nu_;
    rhs_m_.noalias() -= J_ * v;
    solve_constraint_matrix(rhs_m_, mu_);
    v.noalias() += Y_ * mu_;
    rhs_m_ = nu_;
    rhs_m_.noalias() -= J_ * v;
    out.velocity_residual = rhs_m_.norm();
    return out;
}

} // namespace mbd::kernel
