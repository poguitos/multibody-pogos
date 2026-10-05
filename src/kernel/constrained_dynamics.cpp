#include "mbd/kernel/constrained_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include <Eigen/QR>

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
    M_metric_.setZero(nv, nv);
    q_save_.setZero(model.nq);
    q_given_.setZero(model.nq);
    q_best_.setZero(model.nq);
    d_.setZero(nv);
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
    rhs_m_ = gamma_;
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
    if (m_ == 0) {
        ProjectionInfo out;
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

    // Positions, then velocities.
    // Without the least-squares fallback: the dynamics drop the same
    // dependent equations, and the projection agrees with them.
    ProjectionInfo out = gauss_newton(data, q, t, tolerance, max_iterations, nullptr, false);

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

ProjectionInfo ConstraintSolver::gauss_newton(Data& data, VecX& q, Real t, Real tolerance,
                                              int max_iterations, const std::vector<int>* held,
                                              bool least_squares_fallback)
{
    // Steps dq = -W^-1 J^T (J W^-1 J^T)^-1 phi in the metric W factorized in
    // llt_M_: the least change, in that metric, that cancels phi to first
    // order. Each is halved until |phi| decreases, for starts far from the
    // solution.
    ProjectionInfo out;
    evaluate(data, q, zero_v_, t);
    out.position_residual = phi_.norm();
    while (out.position_residual > tolerance && out.iterations < max_iterations) {
        if (held) remove_held_columns(*held);
        factorize_constraint_matrix();
        solve_constraint_matrix(phi_, mu_);
        dv_.setZero();
        add_constraint_motion(mu_, dv_);
        q_save_ = q;
        const Real before = out.position_residual;
        bool decreased = halve_until_decrease(data, q, t, before, out.position_residual);
        if (!decreased && least_squares_fallback && info_.rank < m_) {
            // With dependent equations the step above solves the independent
            // ones and need not reduce |phi| when phi has a part that no
            // motion can cancel (held coordinates that contradict the
            // constraints). The least-squares step always does, unless phi
            // is already as small as the free coordinates can make it.
            least_squares_step();
            decreased = halve_until_decrease(data, q, t, before, out.position_residual);
        }
        ++out.iterations;
        if (!decreased) {
            // No step along this direction reduces |phi|: the iteration has
            // stalled (held coordinates that contradict the constraints, or
            // a configuration where the constraints cannot be met). Keep the
            // best point.
            q = q_save_;
            evaluate(data, q, zero_v_, t);
            out.position_residual = before;
            break;
        }
    }
    out.converged = out.position_residual <= tolerance;
    return out;
}

bool ConstraintSolver::halve_until_decrease(Data& data, VecX& q, Real t, Real before, Real& residual)
{
    Real step = 1.0;
    for (int halving = 0; halving < 12; ++halving) {
        integrate(model_, q_save_, dv_, -step, q);
        evaluate(data, q, zero_v_, t);
        residual = phi_.norm();
        if (residual < before) return true;
        step *= 0.5;
    }
    return false;
}

void ConstraintSolver::least_squares_step()
{
    // dv = W^-1 J^T (J W^-1 J^T)^+ phi with the Moore-Penrose inverse: the
    // least-W-norm step among those that minimize |phi - J dv|. With W = L L^T
    // and Z = L^-1 J^T, dv = L^-T (Z^T)^+ phi. Singular values below
    // sqrt(kRankTolerance) of the largest are dropped, the tolerance the
    // pivots of Z^T Z use. Only reached when an assembly is failing, so it
    // may allocate.
    // The threshold is set before the decomposition, which it shapes.
    Eigen::CompleteOrthogonalDecomposition<MatX> cod;
    cod.setThreshold(std::sqrt(kRankTolerance));
    cod.compute(Z_.transpose());
    dv_ = cod.solve(phi_);
    llt_M_.matrixU().solveInPlace(dv_);
}

void ConstraintSolver::factorize_metric(Data& data, const VecX& q, const std::vector<int>& held)
{
    crba(model_, data, q);
    M_metric_ = data.M;
    for (int k : held) {
        M_metric_.row(k).setZero();
        M_metric_.col(k).setZero();
        M_metric_(k, k) = 1.0;
    }
    llt_M_.compute(M_metric_);
    // llt_M_ no longer holds the mass matrix: project() must not reuse it.
    mass_matrix_factorized_ = false;
}

void ConstraintSolver::remove_held_columns(const std::vector<int>& held)
{
    for (int k : held) J_.col(k).setZero();
}

AssemblyStepInfo ConstraintSolver::assemble_positions(Data& data, VecX& q, Real t,
                                                      const std::vector<int>& held,
                                                      Real tolerance, int max_iterations)
{
    checks::data("kernel::ConstraintSolver::assemble_positions", model_, data);
    checks::q("kernel::ConstraintSolver::assemble_positions", model_, q);
    AssemblyStepInfo out;
    if (m_ == 0) {
        out.converged = true;
        return out;
    }
    // With the held coordinates' rows and columns of the metric decoupled and
    // their columns of J removed, every correction W^-1 J^T mu leaves them
    // exactly as they are.
    factorize_metric(data, q, held);
    q_given_ = q;

    // 1. Reach the constraints.
    const ProjectionInfo reach = gauss_newton(data, q, t, tolerance, max_iterations, &held, true);
    out.iterations = reach.iterations;
    out.residual = reach.position_residual;
    out.converged = reach.converged;
    remove_held_columns(held);
    factorize_constraint_matrix();
    out.rank = info_.rank;
    if (!out.converged) return out;

    // 2. Move along them to the least correction. The least correction d of
    // q_given that satisfies phi = 0 satisfies W d = J^T mu at the solution.
    // Linearizing phi about the current q (correction d_k):
    //     phi + J (d - d_k) = 0,   so   J d = J d_k - phi,
    // whose least-W-norm solution is d = W^-1 J^T (J W^-1 J^T)^-1 (J d_k - phi).
    // From q_given (+) d a Gauss-Newton pass returns to the constraints. The
    // fixed point is the least correction; each refinement moves the
    // correction by a factor of order (curvature x correction) less.
    for (int r = 0; r < max_iterations; ++r) {
        difference(model_, q_given_, q, d_);
        // phi and J are evaluated at q (by gauss_newton), and A factorized
        // from them with the held columns removed.
        rhs_m_ = -phi_;
        rhs_m_.noalias() += J_ * d_;
        solve_constraint_matrix(rhs_m_, mu_);
        dv_.setZero();
        add_constraint_motion(mu_, dv_);
        out.last_change = (dv_ - d_).cwiseAbs().maxCoeff();
        if (out.last_change <= tolerance) break;

        q_best_ = q;
        integrate(model_, q_given_, dv_, 1.0, q);
        const ProjectionInfo back = gauss_newton(data, q, t, tolerance, max_iterations, &held, true);
        ++out.refinements;
        out.iterations += 1 + back.iterations;
        if (!back.converged) {
            // Should not happen near the solution; keep the last point on
            // the constraints.
            q = q_best_;
            evaluate(data, q, zero_v_, t);
            break;
        }
        remove_held_columns(held);
        factorize_constraint_matrix();
    }
    out.residual = phi_.norm();
    out.converged = out.residual <= tolerance;
    return out;
}

AssemblyStepInfo ConstraintSolver::assemble_velocities(Data& data, const VecX& q, VecX& v, Real t,
                                                       const std::vector<int>& held,
                                                       Real tolerance)
{
    checks::data("kernel::ConstraintSolver::assemble_velocities", model_, data);
    checks::q("kernel::ConstraintSolver::assemble_velocities", model_, q);
    checks::v("kernel::ConstraintSolver::assemble_velocities", model_, v);
    AssemblyStepInfo out;
    if (m_ == 0) {
        out.converged = true;
        return out;
    }
    // v += W^-1 J_free^T (J_free W^-1 J_free^T)^-1 (nu - J v): J v = nu with
    // the held velocities unchanged, by the least change to the others.
    factorize_metric(data, q, held);
    evaluate(data, q, zero_v_, t);   // nu does not depend on v
    rhs_m_ = nu_;
    rhs_m_.noalias() -= J_ * v;
    remove_held_columns(held);
    factorize_constraint_matrix();
    out.rank = info_.rank;
    solve_constraint_matrix(rhs_m_, mu_);
    add_constraint_motion(mu_, v);

    evaluate(data, q, zero_v_, t);
    rhs_m_ = nu_;
    rhs_m_.noalias() -= J_ * v;
    out.residual = rhs_m_.norm();
    out.converged = out.residual <= tolerance;
    return out;
}

AssemblyStepInfo ConstraintSolver::assemble_accelerations(Data& data, const VecX& q, const VecX& v,
                                                          VecX& a, Real t,
                                                          const std::vector<int>& held,
                                                          Real tolerance)
{
    checks::data("kernel::ConstraintSolver::assemble_accelerations", model_, data);
    checks::q("kernel::ConstraintSolver::assemble_accelerations", model_, q);
    checks::v("kernel::ConstraintSolver::assemble_accelerations", model_, v);
    checks::v("kernel::ConstraintSolver::assemble_accelerations", model_, a, "a");
    AssemblyStepInfo out;
    if (m_ == 0) {
        out.converged = true;
        return out;
    }
    factorize_metric(data, q, held);
    evaluate(data, q, v, t);   // gamma at (q, v)
    rhs_m_ = gamma_;
    rhs_m_.noalias() -= J_ * a;
    remove_held_columns(held);
    factorize_constraint_matrix();
    out.rank = info_.rank;
    solve_constraint_matrix(rhs_m_, mu_);
    add_constraint_motion(mu_, a);

    evaluate(data, q, v, t);
    rhs_m_ = gamma_;
    rhs_m_.noalias() -= J_ * a;
    out.residual = rhs_m_.norm();
    out.converged = out.residual <= tolerance;
    return out;
}

ProjectionInfo ConstraintSolver::project_positions(Data& data, VecX& q, Real t,
                                                  const std::vector<int>& held,
                                                  Real tolerance, int max_iterations)
{
    checks::data("kernel::ConstraintSolver::project_positions", model_, data);
    checks::q("kernel::ConstraintSolver::project_positions", model_, q);
    if (m_ == 0) {
        ProjectionInfo out;
        out.converged = true;
        return out;
    }
    factorize_metric(data, q, held);
    return gauss_newton(data, q, t, tolerance, max_iterations, &held, true);
}

const VecX& ConstraintSolver::accelerations_at_rest(Data& data, const VecX& q, const VecX& f,
                                                    Real t, const std::vector<int>& held)
{
    checks::data("kernel::ConstraintSolver::accelerations_at_rest", model_, data);
    checks::q("kernel::ConstraintSolver::accelerations_at_rest", model_, q);
    checks::v("kernel::ConstraintSolver::accelerations_at_rest", model_, f, "f");
    // The held coordinates are locked: their rows of W are the identity and
    // their entries of f are dropped (the force that would hold them is not
    // asked for), so their accelerations come out zero.
    factorize_metric(data, q, held);
    rhs_v_ = f;
    for (int k : held) rhs_v_(k) = 0.0;
    v_dot_free_ = rhs_v_;
    llt_M_.solveInPlace(v_dot_free_);
    v_dot_ = v_dot_free_;
    if (m_ == 0) return v_dot_;

    // J a = 0: at rest, with time frozen, gamma has neither velocity products
    // nor the drivers' accelerations.
    evaluate(data, q, zero_v_, t);
    remove_held_columns(held);
    factorize_constraint_matrix();
    rhs_m_.noalias() = J_ * v_dot_free_;
    rhs_m_ = -rhs_m_;
    solve_constraint_matrix(rhs_m_, lambda_);
    add_constraint_motion(lambda_, v_dot_);
    return v_dot_;
}

} // namespace mbd::kernel
