// The generalised-alpha method on the kernel's simulator (plan task 4.3,
// decision D32). Chung and Hulbert (1993), in the form of Arnold and Bruls
// (2007), with positions updated through the joints' exponential maps
// (Bruls, Cardona and Arnold 2012):
//
//     q+ = q (+) (h v + h^2 ((1/2 - beta) a + beta a+))
//     v+ = v + h ((1 - gamma) a + gamma a+)
//     (1 - alpha_m) a+ + alpha_m a = (1 - alpha_f) qdd(q+, v+, x+, t + h) + alpha_f qdd
//
// a is the method's acceleration-like variable, qdd the true acceleration.
// The state elements' states x follow the same formulas a level down, with w
// for a and the rates x_dot for qdd:
//
//     x+ = x + h ((1 - gamma) w + gamma w+)
//     (1 - alpha_m) w+ + alpha_m w = (1 - alpha_f) x_dot(q+, v+, x+, t + h) + alpha_f x_dot
//
// Parameters from rho_infinity: alpha_m = (2 rho - 1) / (rho + 1), alpha_f =
// rho / (rho + 1), gamma = 1/2 - alpha_m + alpha_f, beta = (gamma + 1/2)^2 / 4:
// second order, unconditionally stable for linear systems, with spectral
// radius rho at an infinitely long step. For x' = lambda x the step matrix
// tends, as lambda h -> -infinity, to a lower triangular one with
// eigenvalues 0, -(1 - gamma) / gamma and -rho: x itself is damped to
// O(1 / |lambda h|) each step.

#include "mbd/kernel/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"

namespace mbd::kernel {

namespace {

/// LU factorization of the square A with partial pivoting, in place (unit
/// lower and upper triangles; whole rows swapped, as LAPACK's getrf does).
/// Returns false at a zero pivot. Allocates nothing.
bool lu_factor(MatX& A, std::vector<Index>& pivot)
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
void lu_solve(const MatX& LU, const std::vector<Index>& pivot, VecX& b)
{
    for (Index k = 0; k < LU.rows(); ++k) {
        const Index p = pivot[static_cast<std::size_t>(k)];
        if (p != k) std::swap(b(k), b(p));
    }
    LU.triangularView<Eigen::UnitLower>().solveInPlace(b);
    LU.triangularView<Eigen::Upper>().solveInPlace(b);
}

} // namespace

void Simulator::implicit_jacobian()
{
    const Model& model = system.model;
    const Index nv = model.nv, nx = x.size();
    ++stats_.jacobians;
    // Forward differences from the evaluation at the current state
    // (implicit_qdd_, implicit_xd_); steps of 1e-7, scaled by the entry for
    // velocities and states.
    auto column = [&](Index j, const VecX& qdd, Real eps) {
        implicit_jacobian_.col(j).head(nv) = (qdd - implicit_qdd_) / eps;
        implicit_jacobian_.col(j).tail(nx) = (x_dot_ - implicit_xd_) / eps;
    };
    for (Index j = 0; j < nv; ++j) {
        const Real eps = 1e-7;
        implicit_dq_.setZero();
        implicit_dq_(j) = eps;
        integrate(model, q, implicit_dq_, 1.0, q_stage_);
        column(j, evaluate(q_stage_, v, x, time), eps);
    }
    for (Index j = 0; j < nv; ++j) {
        const Real eps = 1e-7 * std::max(Real(1.0), std::abs(v(j)));
        v_stage_ = v;
        v_stage_(j) += eps;
        column(nv + j, evaluate(q, v_stage_, x, time), eps);
    }
    for (Index j = 0; j < nx; ++j) {
        const Real eps = 1e-7 * std::max(Real(1.0), std::abs(x(j)));
        x_stage_ = x;
        x_stage_(j) += eps;
        column(2 * nv + j, evaluate(q, v, x_stage_, time), eps);
    }
    implicit_jacobian_valid_ = true;
}

void Simulator::step_generalized_alpha(Real h)
{
    const Model& model = system.model;
    const ImplicitControl& ic = implicit_control;
    MBD_THROW_IF(!(ic.rho_infinity >= 0.0 && ic.rho_infinity <= 1.0) || ic.max_iterations < 1,
                 "MBD-K059: kernel::Simulator: implicit_control needs rho_infinity from 0 to 1 and "
                 "max_iterations >= 1; it has "
                     + std::to_string(ic.rho_infinity) + " and " + std::to_string(ic.max_iterations) + ".");
    const Index nv = model.nv, nx = x.size(), n = nv + nx;
    if (implicit_a_.size() != nv || implicit_w_.size() != nx) {
        // First step, or states attached since: size the work (allocates).
        implicit_a_ = VecX::Zero(nv);
        implicit_qdd_ = VecX::Zero(nv);
        implicit_a_new_ = VecX::Zero(nv);
        implicit_dq_ = VecX::Zero(nv);
        implicit_w_ = VecX::Zero(nx);
        implicit_xd_ = VecX::Zero(nx);
        implicit_w_new_ = VecX::Zero(nx);
        implicit_residual_ = VecX::Zero(n);
        implicit_q_ = q;
        implicit_v_ = v;
        implicit_x_ = x;
        implicit_jacobian_ = MatX::Zero(n, 2 * nv + nx);
        implicit_matrix_ = MatX::Zero(n, n);
        implicit_pivots_.assign(static_cast<std::size_t>(n), 0);
        implicit_history_ = false;
        implicit_jacobian_valid_ = false;
    }
    const Real rho = ic.rho_infinity;
    const Real alpha_m = (2.0 * rho - 1.0) / (rho + 1.0);
    const Real alpha_f = rho / (rho + 1.0);
    const Real gamma = 0.5 - alpha_m + alpha_f;
    const Real beta = 0.25 * (gamma + 0.5) * (gamma + 0.5);

    // The accelerations and rates at the start, with the inputs as they are
    // now (tau, the state elements' inputs may have changed since the last
    // step). The method's history belongs to the state its last step left;
    // where anything else has changed the state since (an event's action, the
    // user, a state element's end_step), it starts afresh from here: for q
    // and v together, and for each state on its own, so that a wheel locked
    // by its brake restarts only its own.
    implicit_qdd_ = evaluate(q, v, x, time);
    implicit_xd_ = x_dot_;
    const bool kept = implicit_history_ && time == implicit_time_;
    if (!(kept && q == implicit_q_ && v == implicit_v_)) implicit_a_ = implicit_qdd_;
    for (Index i = 0; i < nx; ++i) {
        if (!kept || x(i) != implicit_x_(i)) implicit_w_(i) = implicit_xd_(i);
    }
    bool fresh = false;
    if (!implicit_jacobian_valid_) {
        implicit_jacobian();
        fresh = true;
    }

    const Real atol = step_control.absolute_tolerance, rtol = step_control.relative_tolerance;
    for (;;) {
        // The iteration matrix for this step length:
        // (1 - alpha_m) I - (1 - alpha_f) [beta h^2 J_q + gamma h J_v, gamma h J_x].
        implicit_matrix_.leftCols(nv) = (beta * h * h) * implicit_jacobian_.leftCols(nv)
                                        + (gamma * h) * implicit_jacobian_.middleCols(nv, nv);
        implicit_matrix_.rightCols(nx) = (gamma * h) * implicit_jacobian_.rightCols(nx);
        implicit_matrix_ *= -(1.0 - alpha_f);
        implicit_matrix_.diagonal().array() += 1.0 - alpha_m;
        const bool regular = lu_factor(implicit_matrix_, implicit_pivots_);

        implicit_a_new_ = implicit_a_;
        implicit_w_new_ = implicit_w_;
        bool converged = false, slow = false;
        Real previous = std::numeric_limits<Real>::infinity();
        for (int k = 1; regular && k <= ic.max_iterations; ++k) {
            implicit_dq_ = h * v + (h * h) * ((0.5 - beta) * implicit_a_ + beta * implicit_a_new_);
            integrate(model, q, implicit_dq_, 1.0, q_stage_);
            v_stage_ = v + h * ((1.0 - gamma) * implicit_a_ + gamma * implicit_a_new_);
            x_stage_ = x + h * ((1.0 - gamma) * implicit_w_ + gamma * implicit_w_new_);
            const VecX& qdd = evaluate(q_stage_, v_stage_, x_stage_, time + h);
            ++stats_.newton_iterations;
            implicit_residual_.head(nv) = -((1.0 - alpha_m) * implicit_a_new_ + alpha_m * implicit_a_
                                            - (1.0 - alpha_f) * qdd - alpha_f * implicit_qdd_);
            implicit_residual_.tail(nx) = -((1.0 - alpha_m) * implicit_w_new_ + alpha_m * implicit_w_
                                            - (1.0 - alpha_f) * x_dot_ - alpha_f * implicit_xd_);
            lu_solve(implicit_matrix_, implicit_pivots_, implicit_residual_);   // now the correction
            implicit_a_new_ += implicit_residual_.head(nv);
            implicit_w_new_ += implicit_residual_.tail(nx);
            // The correction's size in velocities and states, weighed by the
            // tolerances, and never finer than the rounding of the terms the
            // update adds up (64 eps (|v| + h |a|)): on a stiff spring the
            // position update sums terms of h^2 |a| that cancel, and the
            // spring turns their rounding into corrections of about 5 eps h
            // |a|, whatever the stiffness.
            constexpr Real rounding = 64.0 * std::numeric_limits<Real>::epsilon();
            const Real sum =
                ((gamma * h) * implicit_residual_.head(nv).array()
                 / (atol + rtol * v_stage_.array().abs()
                    + rounding * (v_stage_.array().abs() + h * implicit_a_new_.array().abs())))
                    .square()
                    .sum()
                + ((gamma * h) * implicit_residual_.tail(nx).array()
                   / (atol + rtol * x_stage_.array().abs()
                      + rounding * (x_stage_.array().abs() + h * implicit_w_new_.array().abs())))
                      .square()
                      .sum();
            const Real norm = n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : 0.0;
            if (norm <= 1.0) {
                converged = true;
                // Contracting by less than half per iteration: rebuild the
                // matrix before the next step.
                slow = k >= 4 || (k >= 2 && norm > 0.5 * previous);
                break;
            }
            if (!(norm < previous)) break;   // growing, or not a number
            previous = norm;
        }

        if (converged) {
            implicit_dq_ = h * v + (h * h) * ((0.5 - beta) * implicit_a_ + beta * implicit_a_new_);
            integrate(model, q, implicit_dq_, 1.0, q);
            v += h * ((1.0 - gamma) * implicit_a_ + gamma * implicit_a_new_);
            x += h * ((1.0 - gamma) * implicit_w_ + gamma * implicit_w_new_);
            implicit_a_.swap(implicit_a_new_);
            implicit_w_.swap(implicit_w_new_);
            implicit_history_ = true;
            if (slow) implicit_jacobian_valid_ = false;
            return;
        }
        MBD_THROW_IF(fresh, "MBD-K058: kernel::Simulator: the generalised-alpha step of " + std::to_string(h)
                                + " s from t = " + std::to_string(time) + " s did not converge"
                                + (regular ? " within " + std::to_string(ic.max_iterations)
                                                 + " Newton iterations, or its corrections grew, with an "
                                                   "iteration matrix built at its start."
                                           : std::string(": the iteration matrix is singular."))
                                + " The forces change too fast across the step, or are not finite.");
        // A matrix from an earlier state: rebuild it here and try again.
        implicit_jacobian();
        fresh = true;
    }
}

} // namespace mbd::kernel
