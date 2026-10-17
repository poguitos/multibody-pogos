// The implicit methods on the kernel's simulator: generalised-alpha (plan
// task 4.3, decision D32), SDIRK4 and Radau IIA (task 4.4, decisions D33 and
// D34).
//
// Generalised-alpha: Chung and Hulbert (1993), in the form of Arnold and
// Bruls (2007), with positions updated through the joints' exponential maps
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
//
// On a closed mechanism the step is projected onto the constraints afterwards,
// as every method's is, or, with ImplicitControl::stabilized_index2 (task 4.7,
// D37), solved on them in the stabilized index-2 form (Gear, Leimkuhler and
// Gupta 1985; Arnold, Bruls and Cardona 2015): qdd+ gains M^-1 J^T kappa and
// q+'s increment beta h^2 M^-1 J^T eta, and Newton adds phi(q+) = 0 and J v+
// = nu to its equations.

#include "mbd/kernel/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"

#include "dense_lu.hpp"
#include "step_messages.hpp"

namespace mbd::kernel {

namespace {

using Complex = std::complex<Real>;
using detail::lu_factor;
using detail::lu_solve;

/// Whether Newton has stopped making progress: a correction not smaller than
/// the one two iterations before, or not a number. Progress is judged over
/// two iterations because it need not be steady: on the bushed pendulum the
/// corrections fall by a thousand and by two or three in turn, the error
/// passing between two groups of unknowns.
bool newton_stalled(Real norm, Real two_before, int k)
{
    return !std::isfinite(norm) || (k >= 3 && norm >= two_before);
}

/// Whether Newton converged slowly enough that the iteration matrix should
/// be rebuilt: an average contraction worse than one half per iteration.
bool newton_slow(Real first, Real last, int k)
{
    return k >= 2 && std::pow(last / first, 1.0 / static_cast<Real>(k - 1)) > 0.5;
}

/// MBD-K059 for the methods with error control.
void check_iterations(const ImplicitControl& ic)
{
    MBD_THROW_IF(ic.max_iterations < 1, "MBD-K059: kernel::Simulator: implicit_control needs max_iterations >= 1; "
                                        "it has "
                                            + std::to_string(ic.max_iterations) + ".");
}

/// The fraction of the tolerances at which SDIRK4 and Radau IIA stop Newton:
/// RADAU5's min(0.03, sqrt(tol)), tol the larger of the two. Newton's errors
/// add up over the steps undamped while the error control sees only the
/// truncation error, which its third-order estimate overstates; on the
/// bushed pendulum a fixed tenth left Radau IIA's global error at 0.4 to 7
/// times the tolerance, this rule at a hundredth of it (D34).
Real newton_fraction(const StepControl& sc)
{
    return std::min(0.03, std::sqrt(std::max(sc.relative_tolerance, sc.absolute_tolerance)));
}

/// The weight of entry y in Newton's convergence test, for a stage that adds
/// up terms of sizes a and b and whose right-hand side, times the step, turns
/// the rounding of the state into noise of size c / eps: the stopping
/// fraction of the tolerance, but never finer than that rounding. The second
/// part matters beside stiff forces, which multiply the rounding of
/// positions (a bushing of 1e8 N/m turns 1e-16 m into 1e-8 N); without it a
/// tight tolerance asks Newton for corrections below its own noise.
Real newton_weight(const StepControl& sc, Real fraction, Real y, Real a, Real b, Real c)
{
    constexpr Real rounding = 64.0 * std::numeric_limits<Real>::epsilon();
    return fraction * (sc.absolute_tolerance + sc.relative_tolerance * std::abs(y))
           + rounding * (std::abs(a) + std::abs(b) + std::abs(c));
}

// SDIRK4 (Hairer and Wanner, Solving Ordinary Differential Equations II,
// section IV.6, table 6.5; used for multibody dynamics by Haug, Negrut and
// Iancu, 1997): five stages with gamma = 1/4 on the diagonal, stiffly
// accurate (the weights are the last row, so the new state is the last
// stage), L-stable, fourth order; the error estimate is the solution less an
// embedded third-order one (weights b - b_hat). Checked before coding: row
// sums equal the nodes, the eight fourth-order conditions hold, b_hat meets
// the four of third order, |R(iy)| <= 1, R(-infinity) = 0.
constexpr Real sd_gamma = 0.25;
constexpr Real sd_c[5] = {0.25, 0.75, 11.0 / 20.0, 0.5, 1.0};
constexpr Real sd_a[5][5] = {
    {0.25, 0.0, 0.0, 0.0, 0.0},
    {0.5, 0.25, 0.0, 0.0, 0.0},
    {17.0 / 50.0, -1.0 / 25.0, 0.25, 0.0, 0.0},
    {371.0 / 1360.0, -137.0 / 2720.0, 15.0 / 544.0, 0.25, 0.0},
    {25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25}};
constexpr Real sd_e[5] = {-3.0 / 16.0, -27.0 / 32.0, 25.0 / 32.0, 0.0, 0.25};

// Radau IIA with three stages (Hairer and Wanner, section IV.5 and IV.8;
// their code RADAU5): nodes (4 -+ sqrt 6) / 10 and 1, order 5, stage order
// 3, stiffly accurate, L-stable. Its matrix A has one real eigenvalue and a
// complex pair; with A^-1 = T diag(gamma, [alpha -beta; beta alpha]) T^-1 the
// 3n Newton equations of a step split into one real and one complex system of
// n. Checked before coding: row sums equal the nodes, the quadrature
// conditions hold to order 5 and the stage conditions to order 3, R(z) is
// the (2, 3) Pade approximant of e^z, gamma and alpha +- i beta are the roots
// of z^3 - 9 z^2 + 36 z - 60 (the denominator of R).
struct RadauTableau {
    Real c[3];
    Real gamma, alpha, beta;   // the eigenvalues of A^-1: gamma and alpha +- i beta
    Real T[3][3], TI[3][3];    // T and T^-1
    /// The error estimate (RADAU5's): (gamma I - h J)^-1 (h f(y_n) + sum
    /// dd_i Z_i), which is O(h^4).
    Real dd[3];
};

RadauTableau make_radau()
{
    RadauTableau r{};
    const Real s6 = std::sqrt(6.0);
    r.c[0] = (4.0 - s6) / 10.0;
    r.c[1] = (4.0 + s6) / 10.0;
    r.c[2] = 1.0;
    Mat3 A;
    A << (88.0 - 7.0 * s6) / 360.0, (296.0 - 169.0 * s6) / 1800.0, (-2.0 + 3.0 * s6) / 225.0,
        (296.0 + 169.0 * s6) / 1800.0, (88.0 + 7.0 * s6) / 360.0, (-2.0 - 3.0 * s6) / 225.0,
        (16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0;
    const Mat3 Ainv = A.inverse();
    // The roots of z^3 - 9 z^2 + 36 z - 60 in closed form.
    const Real a81 = std::cbrt(81.0), a9 = std::cbrt(9.0);
    r.gamma = 30.0 / (6.0 + a81 - a9);
    const Real ar = (12.0 - a81 + a9) / 60.0, ai = (a81 + a9) * std::sqrt(3.0) / 60.0;   // eigenvalues of A
    r.alpha = ar / (ar * ar + ai * ai);
    r.beta = ai / (ar * ar + ai * ai);
    // Eigenvectors: a null vector of A^-1 - mu I is the cross product of two
    // of its rows (without conjugation for the complex one); the pair with
    // the largest product.
    auto null_vector = [](const auto& M) {
        using V = Eigen::Matrix<typename std::decay_t<decltype(M)>::Scalar, 3, 1>;
        auto cross = [](const V& a, const V& b) {
            return V(a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0));
        };
        V best = cross(M.row(0).transpose(), M.row(1).transpose());
        for (const auto [i, j] : {std::pair<int, int>{0, 2}, std::pair<int, int>{1, 2}}) {
            const V u = cross(M.row(i).transpose(), M.row(j).transpose());
            if (u.norm() > best.norm()) best = u;
        }
        return best;
    };
    const Vec3 v1 = null_vector(Mat3(Ainv - r.gamma * Mat3::Identity()));
    const Eigen::Vector3cd u =
        null_vector(Eigen::Matrix3cd(Ainv.cast<Complex>() - Complex(r.alpha, r.beta) * Eigen::Matrix3cd::Identity()));
    // A^-1 (Re u) = alpha Re u - beta Im u and A^-1 (Im u) = beta Re u + alpha
    // Im u, so the columns v1, Re u, -Im u give the block [alpha -beta; beta
    // alpha].
    Mat3 T;
    T.col(0) = v1;
    T.col(1) = u.real();
    T.col(2) = -u.imag();
    const Mat3 TI = T.inverse();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.T[i][j] = T(i, j);
            r.TI[i][j] = TI(i, j);
        }
    }
    r.dd[0] = -(13.0 + 7.0 * s6) / 3.0;
    r.dd[1] = (-13.0 + 7.0 * s6) / 3.0;
    r.dd[2] = -1.0 / 3.0;
    return r;
}

const RadauTableau& radau()
{
    static const RadauTableau tableau = make_radau();
    return tableau;
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
    // On the constraints (D37), also the derivative G of J v - nu with
    // respect to q, from the same evaluations: the solver has J and nu at
    // each.
    const bool constrained = implicit_G_.rows() > 0;
    if (constrained) {
        solver_.evaluate(data_, q, v, time);
        implicit_g0_.noalias() = solver_.J() * v;
        implicit_g0_ -= solver_.nu();
    }
    for (Index j = 0; j < nv; ++j) {
        const Real eps = 1e-7;
        implicit_dq_.setZero();
        implicit_dq_(j) = eps;
        integrate(model, q, implicit_dq_, 1.0, q_stage_);
        column(j, evaluate(q_stage_, v, x, time), eps);
        if (constrained) {
            drift_work_.noalias() = solver_.J() * v;
            drift_work_ -= solver_.nu();
            implicit_G_.col(j) = (drift_work_ - implicit_g0_) / eps;
        }
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
    // On the constraints (D37): a pair of extra unknowns per equation, kappa
    // and eta, those of redundant equations left out; eta is eliminated from
    // the Newton system, which keeps n + m unknowns.
    const Index m = stabilized_alpha() ? solver_.size() : 0;
    const Index N = n + m;
    if (implicit_a_.size() != nv || implicit_w_.size() != nx || implicit_matrix_.rows() != N) {
        // First step, states attached since, or the constraints switched on
        // or off: size the work (allocates).
        implicit_a_ = VecX::Zero(nv);
        implicit_qdd_ = VecX::Zero(nv);
        implicit_qddk_ = VecX::Zero(nv);
        implicit_qdd_plus_ = VecX::Zero(nv);
        implicit_a_new_ = VecX::Zero(nv);
        implicit_dq_ = VecX::Zero(nv);
        implicit_w_ = VecX::Zero(nx);
        implicit_xd_ = VecX::Zero(nx);
        implicit_w_new_ = VecX::Zero(nx);
        implicit_residual_ = VecX::Zero(N);
        implicit_q_ = q;
        implicit_v_ = v;
        implicit_x_ = x;
        implicit_jacobian_ = MatX::Zero(n, 2 * nv + nx);
        implicit_G_ = MatX::Zero(m, nv);
        implicit_g0_ = VecX::Zero(m);
        implicit_matrix_ = MatX::Zero(N, N);
        implicit_pivots_.assign(static_cast<std::size_t>(N), 0);
        implicit_rows_.assign(static_cast<std::size_t>(m), 0);
        implicit_Jc_ = MatX::Zero(m, nv);
        implicit_P_ = MatX::Zero(nv, m);
        implicit_Gc_ = MatX::Zero(m, nv);
        implicit_A_ = MatX::Zero(m, m);
        implicit_llt_ = Eigen::LLT<MatX>(m);
        implicit_F_ = MatX::Zero(m, n);
        implicit_JqP_ = MatX::Zero(n, m);
        implicit_GP_ = MatX::Zero(m, m);
        implicit_rp_ = VecX::Zero(m);
        implicit_f_ = VecX::Zero(m);
        implicit_deta_ = VecX::Zero(m);
        implicit_kappa_ = VecX::Zero(m);
        implicit_kappa_new_ = VecX::Zero(m);
        implicit_eta_new_ = VecX::Zero(m);
        implicit_unit_ = VecX::Zero(m);
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
    const bool kept_motion = kept && q == implicit_q_ && v == implicit_v_;
    for (Index i = 0; i < nx; ++i) {
        if (!kept || x(i) != implicit_x_(i)) implicit_w_(i) = implicit_xd_(i);
    }
    // On the constraints, the accelerations of the method's relation at the
    // start are those the last step ended with, its extra constraint forces J^T
    // kappa included.
    implicit_qddk_ = implicit_qdd_;
    if (m > 0) {
        if (kept_motion) {
            solver_.add_constraint_motion(implicit_kappa_, implicit_qddk_);
        } else {
            implicit_kappa_.setZero();
        }
    }
    if (!kept_motion) implicit_a_ = implicit_qddk_;
    bool fresh = false;
    if (!implicit_jacobian_valid_) {
        implicit_jacobian();
        fresh = true;
    }

    const Real atol = step_control.absolute_tolerance, rtol = step_control.relative_tolerance;
    const Real c = 1.0 - alpha_f;
    for (;;) {
        implicit_a_new_ = implicit_a_;
        implicit_w_new_ = implicit_w_;
        implicit_eta_new_.setZero();
        int rank = 0;
        bool regular = true, converged = false, slow = false;
        Real first = 0.0, previous = std::numeric_limits<Real>::infinity(), before = previous;
        for (int k = 1; k <= ic.max_iterations; ++k) {
            implicit_dq_ = h * v + (h * h) * ((0.5 - beta) * implicit_a_ + beta * implicit_a_new_);
            if (k > 1 && m > 0) implicit_dq_.noalias() += (beta * h * h) * (implicit_P_ * implicit_eta_new_);
            integrate(model, q, implicit_dq_, 1.0, q_stage_);
            v_stage_ = v + h * ((1.0 - gamma) * implicit_a_ + gamma * implicit_a_new_);
            x_stage_ = x + h * ((1.0 - gamma) * implicit_w_ + gamma * implicit_w_new_);
            implicit_qdd_plus_ = evaluate(q_stage_, v_stage_, x_stage_, time + h);
            ++stats_.newton_iterations;
            if (m > 0 && k == 1) {
                // At the predictor, near the step's end: the independent
                // equations, in the order the rank-revealing factorization
                // took them, their rows of J and the columns of M^-1 J^T,
                // kept for the step. The residuals use the current J, so the
                // solution does not depend on them; the closer they are to
                // the end, the faster Newton converges. The multipliers start
                // from the last step's.
                rank = solver_.info().rank;
                implicit_Jc_.setZero();
                implicit_Gc_.setZero();
                implicit_P_.setZero();
                implicit_kappa_new_.setZero();
                for (Index i = 0; i < m; ++i) {
                    implicit_rows_[static_cast<std::size_t>(i)] =
                        i < rank ? solver_.independent_equation(static_cast<int>(i)) : -1;
                }
                for (Index i = 0; i < rank; ++i) {
                    const Index row = implicit_rows_[static_cast<std::size_t>(i)];
                    implicit_kappa_new_(i) = implicit_kappa_(row);
                    implicit_Jc_.row(i) = solver_.J().row(row);
                    implicit_Gc_.row(i) = implicit_G_.row(row);
                    implicit_unit_.setZero();
                    implicit_unit_(row) = 1.0;
                    implicit_dq_.setZero();
                    solver_.add_constraint_motion(implicit_unit_, implicit_dq_);
                    implicit_P_.col(i) = implicit_dq_;
                }
            }
            if (k == 1) {
                // The iteration matrix for this step length. On the unknowns
                // z = (a+, w+) and, on the constraints, kappa and eta, with
                // S = (1 - alpha_m) I - c [beta h^2 J_q + gamma h J_v, gamma h
                // J_x]:
                //
                //   [ S                  -c P   -c beta h^2 J_q P ]   z       rows of the method's relation
                //   [ J_c                 0      A                ]   kappa   phi(q+) = 0, scaled by 1 / (beta h^2)
                //   [ J_c + g G_c         0      g G_c P          ]   eta     J v+ = nu, scaled by 1 / (gamma h)
                //
                // P = M^-1 J_c^T, A = J_c P (positive definite: independent
                // equations), g = beta h / gamma, G = d(J v - nu)/dq from the
                // iteration matrix's differences; eta is scaled by beta h^2,
                // so that every block is of the size of an acceleration. The
                // derivatives of M^-1 J^T kappa with respect to q are left
                // out (kappa is small): a simplified Newton's method, the
                // residuals exact. The second row gives eta = f - F z, with
                // F = A^-1 [J_c 0] and f = A^-1 times its right-hand side,
                // which leaves, on (z, kappa),
                //
                //   [ S + c beta h^2 (J_q P) F               -c P ]
                //   [ [J_c + g G_c, 0] - g (G_c P) F           0  ]
                //
                // n + m square instead of n + 2 m: on the sedan (n = 31, m =
                // 17) a third of the LU's work.
                implicit_matrix_.setZero();
                auto S = implicit_matrix_.topLeftCorner(n, n);
                S.leftCols(nv) = (beta * h * h) * implicit_jacobian_.leftCols(nv)
                                 + (gamma * h) * implicit_jacobian_.middleCols(nv, nv);
                S.rightCols(nx) = (gamma * h) * implicit_jacobian_.rightCols(nx);
                S *= -c;
                S.diagonal().array() += 1.0 - alpha_m;
                if (m > 0) {
                    const Real g = beta / gamma * h;
                    // A, its factor and F; a redundant equation's row and
                    // column of A are the identity's, its rows of J_c, G_c and
                    // F zero.
                    implicit_A_.noalias() = implicit_Jc_ * implicit_P_;
                    for (Index i = rank; i < m; ++i) implicit_A_(i, i) = 1.0;
                    implicit_llt_.compute(implicit_A_);
                    if (implicit_llt_.info() != Eigen::Success) {
                        regular = false;
                        break;
                    }
                    implicit_F_.setZero();
                    implicit_F_.leftCols(nv) = implicit_Jc_;
                    implicit_llt_.solveInPlace(implicit_F_);
                    implicit_JqP_.noalias() = implicit_jacobian_.leftCols(nv) * implicit_P_;
                    implicit_GP_.noalias() = implicit_Gc_ * implicit_P_;
                    S.noalias() += (c * beta * h * h) * (implicit_JqP_ * implicit_F_);
                    implicit_matrix_.topRightCorner(nv, m) = -c * implicit_P_;
                    auto V = implicit_matrix_.bottomLeftCorner(m, n);
                    V.leftCols(nv) = implicit_Jc_ + g * implicit_Gc_;
                    V.noalias() -= g * (implicit_GP_ * implicit_F_);
                    // A redundant equation: a dummy kappa, held at zero.
                    for (Index i = rank; i < m; ++i) implicit_matrix_(n + i, n + i) = 1.0;
                }
                regular = lu_factor(implicit_matrix_, implicit_pivots_);
                if (!regular) break;
            }
            Real constraint_norm = 0.0;
            if (m > 0) {
                // The extra constraint forces at the new state, and the
                // constraints there (every equation; the dummies' rows zero).
                implicit_unit_.setZero();
                for (Index i = 0; i < rank; ++i) {
                    implicit_unit_(implicit_rows_[static_cast<std::size_t>(i)]) = implicit_kappa_new_(i);
                }
                solver_.add_constraint_motion(implicit_unit_, implicit_qdd_plus_);
                drift_work_.noalias() = solver_.J() * v_stage_;
                drift_work_ -= solver_.nu();
                implicit_rp_.setZero();
                implicit_residual_.tail(m).setZero();
                for (Index i = 0; i < rank; ++i) {
                    const Index row = implicit_rows_[static_cast<std::size_t>(i)];
                    implicit_rp_(i) = -solver_.phi()(row) / (beta * h * h);
                    implicit_residual_(n + i) = -drift_work_(row) / (gamma * h);
                }
                // Positions to constraint_tolerance; J v - nu to the
                // tolerance the velocities themselves are held to (a
                // residual there does not add up: the next step's positions
                // are held to phi = 0 again). Asked for 1e-10, it cost two
                // more iterations a step, and the last correction, through
                // the simplified matrix, left 4e-9 all the same.
                constraint_norm =
                    std::max(solver_.phi().norm() / constraint_tolerance, drift_work_.norm() / atol);
            }
            implicit_residual_.head(nv) = -((1.0 - alpha_m) * implicit_a_new_ + alpha_m * implicit_a_
                                            - c * implicit_qdd_plus_ - alpha_f * implicit_qddk_);
            implicit_residual_.segment(nv, nx) = -((1.0 - alpha_m) * implicit_w_new_ + alpha_m * implicit_w_
                                                   - c * x_dot_ - alpha_f * implicit_xd_);
            if (m > 0) {
                // eta eliminated (see the matrix): f = A^-1 times the scaled
                // phi's right-hand side, moved into the other rows.
                implicit_f_ = implicit_rp_;
                implicit_llt_.solveInPlace(implicit_f_);
                implicit_residual_.head(n).noalias() += (c * beta * h * h) * (implicit_JqP_ * implicit_f_);
                implicit_residual_.tail(m).noalias() -= (beta / gamma * h) * (implicit_GP_ * implicit_f_);
            }
            lu_solve(implicit_matrix_, implicit_pivots_, implicit_residual_);   // now the correction
            implicit_a_new_ += implicit_residual_.head(nv);
            implicit_w_new_ += implicit_residual_.segment(nv, nx);
            if (m > 0) {
                implicit_kappa_new_ += implicit_residual_.tail(m);
                implicit_deta_ = implicit_f_;
                implicit_deta_.noalias() -= implicit_F_ * implicit_residual_.head(n);
                implicit_eta_new_ += implicit_deta_;
            }
            // The correction's size in velocities and states, weighed by the
            // tolerances, and never finer than the rounding of the terms the
            // update adds up (64 eps (|v| + h |a|)): on a stiff spring the
            // position update sums terms of h^2 |a| that cancel, and the
            // spring turns their rounding into corrections of about 5 eps h
            // |a|, whatever the stiffness. On the constraints, also their
            // residuals at the state just evaluated (above).
            constexpr Real rounding = 64.0 * std::numeric_limits<Real>::epsilon();
            const Real sum =
                ((gamma * h) * implicit_residual_.head(nv).array()
                 / (atol + rtol * v_stage_.array().abs()
                    + rounding * (v_stage_.array().abs() + h * implicit_a_new_.array().abs())))
                    .square()
                    .sum()
                + ((gamma * h) * implicit_residual_.segment(nv, nx).array()
                   / (atol + rtol * x_stage_.array().abs()
                      + rounding * (x_stage_.array().abs() + h * implicit_w_new_.array().abs())))
                      .square()
                      .sum();
            const Real norm = std::max(n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : 0.0, constraint_norm);
            if (k == 1) first = norm;
            if (norm <= 1.0) {
                converged = true;
                slow = newton_slow(first, norm, k);   // then rebuild the matrix before the next step
                break;
            }
            if (newton_stalled(norm, before, k)) break;
            before = previous;
            previous = norm;
        }

        if (converged) {
            implicit_dq_ = h * v + (h * h) * ((0.5 - beta) * implicit_a_ + beta * implicit_a_new_);
            if (m > 0) implicit_dq_.noalias() += (beta * h * h) * (implicit_P_ * implicit_eta_new_);
            integrate(model, q, implicit_dq_, 1.0, q);
            v += h * ((1.0 - gamma) * implicit_a_ + gamma * implicit_a_new_);
            x += h * ((1.0 - gamma) * implicit_w_ + gamma * implicit_w_new_);
            implicit_a_.swap(implicit_a_new_);
            implicit_w_.swap(implicit_w_new_);
            if (m > 0) {
                // The constraint forces by equation, for the next step.
                implicit_kappa_.setZero();
                for (Index i = 0; i < rank; ++i) {
                    implicit_kappa_(implicit_rows_[static_cast<std::size_t>(i)]) = implicit_kappa_new_(i);
                }
            }
            implicit_history_ = true;
            if (slow) implicit_jacobian_valid_ = false;
            return;
        }
        if (fresh) detail::throw_unsolved("generalised-alpha", h, time, regular, ic.max_iterations);
        // A matrix from an earlier state: rebuild it here and try again.
        implicit_jacobian();
        fresh = true;
    }
}

// --- The stacked first-order system --------------------------------------------
//
// SDIRK4 and Radau IIA integrate y = (q, v, x), y' = f(y) = (q_dot(q, v),
// accelerations, rates), q in its embedding (quaternions as four numbers,
// normalized once per step, as RK4 and Dormand-Prince do; updating through
// the exponential map would need further terms beyond second order). Both
// solve their stages by Newton's method on matrices built from J = df/dy,
// kept from step to step: J costs n + 1 evaluations, so it is built again
// only when the Newton iterations a step costs, beyond those of the first
// step with it, exceed that (and wherever Newton fails).

void Simulator::size_stacked(Index n)
{
    if (stacked_jacobian_.rows() == n && stacked_qd_.size() == system.model.nq) return;
    stacked_jacobian_ = MatX::Zero(n, n);
    stacked_f0_ = VecX::Zero(n);
    stacked_y_ = VecX::Zero(n);
    stacked_fp_ = VecX::Zero(n);
    stacked_qd_ = VecX::Zero(system.model.nq);
    stacked_noise_ = VecX::Zero(n);
    stacked_jacobian_valid_ = false;
    ++stacked_version_;
}

void Simulator::stacked_f(const VecX& y, Real t, VecX& out)
{
    const Model& model = system.model;
    const Index nq = model.nq, nv = model.nv, nx = x.size();
    q_stage_ = y.head(nq);
    v_stage_ = y.segment(nq, nv);
    x_stage_ = y.tail(nx);
    out.segment(nq, nv) = evaluate(q_stage_, v_stage_, x_stage_, t);
    out.tail(nx) = x_dot_;
    q_dot(model, q_stage_, v_stage_, stacked_qd_);
    out.head(nq) = stacked_qd_;
}

void Simulator::stacked_jacobian(const VecX& y, Real t)
{
    ++stats_.jacobians;
    stacked_f(y, t, stacked_f0_);
    stacked_y_ = y;
    for (Index j = 0; j < y.size(); ++j) {
        // Steps of 1e-7, scaled by the entry; divided by the step actually
        // taken after rounding.
        const Real keep = stacked_y_(j);
        stacked_y_(j) = keep + 1e-7 * std::max(Real(1.0), std::abs(keep));
        const Real eps = stacked_y_(j) - keep;
        stacked_f(stacked_y_, t, stacked_fp_);
        stacked_jacobian_.col(j) = (stacked_fp_ - stacked_f0_) / eps;
        stacked_y_(j) = keep;
    }
    stacked_jacobian_valid_ = true;
    ++stacked_version_;
    stacked_base_cost_ = -1;
}

void Simulator::stacked_jacobian_policy(int evaluations)
{
    if (stacked_base_cost_ < 0) {
        stacked_base_cost_ = evaluations;
    } else if (evaluations - stacked_base_cost_ > stacked_jacobian_.rows() + 1) {
        stacked_jacobian_valid_ = false;
    }
}

void Simulator::stacked_noise(const VecX& y)
{
    for (Index i = 0; i < y.size(); ++i) {
        Real s = 0.0;
        for (Index j = 0; j < y.size(); ++j) s += std::abs(stacked_jacobian_(i, j) * y(j));
        stacked_noise_(i) = s;
    }
}

// --- SDIRK4 --------------------------------------------------------------------
//
// Stage i:
//
//     Y_i = S_i + gamma h f(Y_i),   S_i = y_n + h sum_{j<i} a_ij K_j,
//
// solved by Newton with the matrix I - gamma h J, the same for every stage;
// then K_i = (Y_i - S_i) / (gamma h), which spares an evaluation. The new
// state is Y_5. The error estimate h sum (b_j - b_hat_j) K_j is filtered
// through (I - gamma h J)^-1 (Shampine's device): the embedded solution is not
// A-stable (|R_hat(iy)| reaches 3.3), and unfiltered its estimate would grow
// with the stiff components the method itself damps.
//
// Newton stops when its correction falls below a fraction of the tolerance
// (newton_fraction), and fails when a correction is not smaller than the one
// two iterations before or the iterations run out; a stage that fails is
// solved again with J built at its predictor. Hairer and Wanner's tests on the rate of contraction
// (stop when theta / (1 - theta) times the correction is small enough; fail
// early when the iterations left are not expected to suffice) were tried and
// dropped: on the bushed pendulum the first correction from the predictor
// contracts far less than those after it, so they gave up on solves that
// were about to converge, and built J again almost every stage.

bool Simulator::sdirk_factor(Real h)
{
    if (sdirk_factored_h_ == h && sdirk_factored_version_ == stacked_version_) return sdirk_factored_regular_;
    sdirk_matrix_ = (-sd_gamma * h) * stacked_jacobian_;
    sdirk_matrix_.diagonal().array() += 1.0;
    sdirk_factored_regular_ = lu_factor(sdirk_matrix_, sdirk_pivots_);
    sdirk_factored_h_ = h;
    sdirk_factored_version_ = stacked_version_;
    return sdirk_factored_regular_;
}

bool Simulator::sdirk_newton(int i, Real h, int& iterations)
{
    const Index n = sdirk_y_.size();
    const Real gh = sd_gamma * h, t = time + sd_c[i] * h, fraction = newton_fraction(step_control);
    Real d1 = 0.0, d2 = 0.0;   // the sizes of the last two corrections
    for (int k = 1; k <= implicit_control.max_iterations; ++k) {
        stacked_f(sdirk_y_, t, sdirk_r_);
        ++stats_.newton_iterations;
        ++iterations;
        sdirk_r_ = sdirk_s_ + gh * sdirk_r_ - sdirk_y_;
        lu_solve(sdirk_matrix_, sdirk_pivots_, sdirk_r_);   // now the correction
        sdirk_y_ += sdirk_r_;
        Real sum = 0.0;
        for (Index j = 0; j < n; ++j) {
            const Real s = sdirk_r_(j) / newton_weight(step_control, fraction, sdirk_y_(j), sdirk_s_(j),
                                                         sdirk_y_(j) - sdirk_s_(j), gh * stacked_noise_(j));
            sum += s * s;
        }
        const Real d = n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : 0.0;
        if (d <= 1.0) return true;
        if (newton_stalled(d, d2, k)) return false;
        d2 = d1;
        d1 = d;
    }
    return false;
}

bool Simulator::sdirk_stages(Real h, int& iterations, bool& rebuilt, bool& singular)
{
    const Real gh = sd_gamma * h;
    for (int i = 0; i < 5; ++i) {
        sdirk_s_ = sdirk_yn_;
        for (int j = 0; j < i; ++j) sdirk_s_ += (h * sd_a[i][j]) * sdirk_k_[j];
        for (int attempt = 0;; ++attempt) {
            // Predictor: the stage before's derivative (for the first stage,
            // f at the step's start, when known).
            sdirk_y_ = sdirk_s_;
            if (i > 0) sdirk_y_ += gh * sdirk_k_[i - 1];
            else if (sdirk_fn_valid_) sdirk_y_ += gh * sdirk_fn_;
            if (attempt == 1) {
                // J no longer serves here: build it at this stage.
                stacked_jacobian(sdirk_y_, time + sd_c[i] * h);
                rebuilt = true;
                stacked_noise(sdirk_yn_);
                if (!sdirk_factor(h)) {
                    singular = true;
                    return false;
                }
            }
            if (sdirk_newton(i, h, iterations)) break;
            if (attempt == 1) return false;
        }
        sdirk_k_[i] = (sdirk_y_ - sdirk_s_) / gh;
    }
    return true;
}

void Simulator::step_sdirk(Real dt)
{
    const Model& model = system.model;
    const StepControl& sc = step_control;
    detail::check_step_control(sc);
    check_iterations(implicit_control);
    const Index nq = model.nq, nv = model.nv, nx = x.size(), n = nq + nv + nx;
    size_stacked(n);
    if (sdirk_yn_.size() != n) {
        // First step, or states attached since: size the work (allocates).
        sdirk_matrix_ = MatX::Zero(n, n);
        sdirk_pivots_.assign(static_cast<std::size_t>(n), 0);
        sdirk_factored_version_ = -1;
        for (auto& k : sdirk_k_) k = VecX::Zero(n);
        sdirk_yn_ = VecX::Zero(n);
        sdirk_s_ = VecX::Zero(n);
        sdirk_y_ = VecX::Zero(n);
        sdirk_fn_ = VecX::Zero(n);
        sdirk_r_ = VecX::Zero(n);
        sdirk_e_ = VecX::Zero(n);
        sdirk_fn_valid_ = false;
    }

    const Real t_end = time + dt;
    Real h = sc.adaptive && next_step_ > 0.0 ? std::min(next_step_, dt) : dt;
    bool after_rejection = false;
    int tried = 0;
    while (time < t_end) {
        if (++tried > sc.max_steps) detail::throw_too_many_steps("SDIRK4", sc.max_steps, dt, time);
        const Real remaining = t_end - time;
        const bool last = 1.01 * h >= remaining;
        const Real step = last ? remaining : h;

        sdirk_yn_.head(nq) = q;
        sdirk_yn_.segment(nq, nv) = v;
        sdirk_yn_.tail(nx) = x;
        if (!stacked_jacobian_valid_) {
            stacked_jacobian(sdirk_yn_, time);
            sdirk_fn_ = stacked_f0_;   // f at the step's start, exactly
            sdirk_fn_valid_ = true;
        }
        stacked_noise(sdirk_yn_);
        int iterations = 0;
        bool rebuilt = false, singular = !sdirk_factor(step);
        if (singular || !sdirk_stages(step, iterations, rebuilt, singular)) {
            // Newton failed even with a matrix built where it did.
            if (!sc.adaptive) {
                detail::throw_unsolved("SDIRK4", step, time, !singular, implicit_control.max_iterations);
            }
            ++stats_.rejected;
            h = 0.25 * step;
            if (h < sc.min_step || time + h == time) {
                detail::throw_step_too_short("SDIRK4", h, time, sc.min_step, std::numeric_limits<Real>::infinity());
            }
            after_rejection = true;
            continue;
        }

        // The error estimate, filtered, weighed as Dormand-Prince's.
        sdirk_e_.setZero();
        for (int j = 0; j < 5; ++j) {
            if (sd_e[j] != 0.0) sdirk_e_ += (step * sd_e[j]) * sdirk_k_[j];
        }
        lu_solve(sdirk_matrix_, sdirk_pivots_, sdirk_e_);
        const Real atol = sc.absolute_tolerance, rtol = sc.relative_tolerance;
        const Real sum =
            (sdirk_e_.array() / (atol + rtol * sdirk_yn_.array().abs().max(sdirk_y_.array().abs()))).square().sum();
        const Real error = n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : 0.0;
        if (sc.adaptive && !(error <= 1.0)) {
            ++stats_.rejected;
            h = step * (std::isfinite(error) ? std::max(0.2, 0.9 * std::pow(error, -0.25)) : 0.2);
            if (h < sc.min_step || time + h == time) detail::throw_step_too_short("SDIRK4", h, time, sc.min_step, error);
            after_rejection = true;
            continue;
        }

        // Accepted: the new state is the last stage, and its derivative the
        // next step's first predictor.
        q = sdirk_y_.head(nq);
        v = sdirk_y_.segment(nq, nv);
        x = sdirk_y_.tail(nx);
        normalize(model, q);
        sdirk_fn_ = sdirk_k_[4];
        time = last ? t_end : time + step;
        ++stats_.steps;
        stats_.step_size = step;
        stats_.error = error;
        if (sc.adaptive) {
            const Real grow = after_rejection ? 1.0 : 5.0;
            const Real proposal =
                step * (error > 0.0 ? std::min(grow, std::max(0.2, 0.9 * std::pow(error, -0.25))) : grow);
            h = last && step < h ? std::max(h, proposal) : proposal;
        }
        after_rejection = false;
        // A J built within the step: its first whole step is the next.
        if (!rebuilt) stacked_jacobian_policy(iterations);
        project_step();
    }
    next_step_ = h;
}

// --- Radau IIA -----------------------------------------------------------------
//
// The stages' increments Z_i = Y_i - y_n solve Z = h (A x I) F(Z), F_i =
// f(y_n + Z_i). Simplified Newton on (A^-1 x I) Z / h - F(Z) = 0, with every
// F_i's derivative replaced by J, transformed by W = (T^-1 x I) Z:
//
//     (gamma I - h J) dW_1 = h G_1 - gamma W_1
//     ((alpha + i beta) I - h J) (dW_2 + i dW_3) = h (G_2 + i G_3) - (alpha + i beta)(W_2 + i W_3)
//
// with G = (T^-1 x I) F: one real and one complex LU of n, against one of 3n.
// A Newton iteration costs three evaluations. The new state is y_n + Z_3.
// Starting values come from the last step's collocation polynomial, carried
// on to the new stages' times; the error estimate is RADAU5's (see
// RadauTableau), evaluated again at y_n + estimate after a first step or a
// rejection, when stiff components far from equilibrium would otherwise
// inflate it (Hairer and Wanner, section IV.8).

bool Simulator::radau_factor(Real h)
{
    if (radau_factored_h_ == h && radau_factored_version_ == stacked_version_) return radau_factored_regular_;
    const RadauTableau& rt = radau();
    radau_e1_ = (-h) * stacked_jacobian_;
    radau_e1_.diagonal().array() += rt.gamma;
    radau_e2_ = ((-h) * stacked_jacobian_).cast<Complex>();
    radau_e2_.diagonal().array() += Complex(rt.alpha, rt.beta);
    radau_factored_regular_ = lu_factor(radau_e1_, radau_p1_) && lu_factor(radau_e2_, radau_p2_);
    radau_factored_h_ = h;
    radau_factored_version_ = stacked_version_;
    return radau_factored_regular_;
}

bool Simulator::radau_newton(Real h, int& iterations)
{
    const RadauTableau& rt = radau();
    const Index n = radau_yn_.size();
    const Real fraction = newton_fraction(step_control);
    for (int i = 0; i < 3; ++i) {
        radau_w_[i] = rt.TI[i][0] * radau_z_[0] + rt.TI[i][1] * radau_z_[1] + rt.TI[i][2] * radau_z_[2];
    }
    Real d1 = 0.0, d2 = 0.0;   // the sizes of the last two corrections
    for (int k = 1; k <= implicit_control.max_iterations; ++k) {
        for (int i = 0; i < 3; ++i) {
            radau_y_ = radau_yn_ + radau_z_[i];
            stacked_f(radau_y_, time + rt.c[i] * h, radau_f_[i]);
        }
        ++stats_.newton_iterations;
        ++iterations;
        radau_r_ = h * (rt.TI[0][0] * radau_f_[0] + rt.TI[0][1] * radau_f_[1] + rt.TI[0][2] * radau_f_[2])
                   - rt.gamma * radau_w_[0];
        for (Index j = 0; j < n; ++j) {
            const Real g2 = h * (rt.TI[1][0] * radau_f_[0](j) + rt.TI[1][1] * radau_f_[1](j) + rt.TI[1][2] * radau_f_[2](j));
            const Real g3 = h * (rt.TI[2][0] * radau_f_[0](j) + rt.TI[2][1] * radau_f_[1](j) + rt.TI[2][2] * radau_f_[2](j));
            const Real w2 = radau_w_[1](j), w3 = radau_w_[2](j);
            radau_rc_(j) = Complex(g2 - (rt.alpha * w2 - rt.beta * w3), g3 - (rt.beta * w2 + rt.alpha * w3));
        }
        lu_solve(radau_e1_, radau_p1_, radau_r_);
        lu_solve(radau_e2_, radau_p2_, radau_rc_);
        // Apply the corrections dW and dZ = (T x I) dW, and weigh dZ as
        // SDIRK4 weighs its stages' corrections.
        Real sum = 0.0;
        for (Index j = 0; j < n; ++j) {
            const Real dw[3] = {radau_r_(j), radau_rc_(j).real(), radau_rc_(j).imag()};
            for (int i = 0; i < 3; ++i) {
                radau_w_[i](j) += dw[i];
                const Real dz = rt.T[i][0] * dw[0] + rt.T[i][1] * dw[1] + rt.T[i][2] * dw[2];
                const Real z = radau_z_[i](j) += dz;
                const Real y0 = radau_yn_(j);
                const Real scale = newton_weight(step_control, fraction, y0 + z, y0, z, h * stacked_noise_(j));
                sum += (dz / scale) * (dz / scale);
            }
        }
        const Real d = n > 0 ? std::sqrt(sum / static_cast<Real>(3 * n)) : 0.0;
        if (d <= 1.0) return true;
        if (newton_stalled(d, d2, k)) return false;
        d2 = d1;
        d1 = d;
    }
    return false;
}

void Simulator::step_radau(Real dt)
{
    const Model& model = system.model;
    const StepControl& sc = step_control;
    detail::check_step_control(sc);
    check_iterations(implicit_control);
    const RadauTableau& rt = radau();
    const Index nq = model.nq, nv = model.nv, nx = x.size(), n = nq + nv + nx;
    size_stacked(n);
    if (radau_yn_.size() != n) {
        // First step, or states attached since: size the work (allocates).
        radau_e1_ = MatX::Zero(n, n);
        radau_e2_ = Eigen::MatrixXcd::Zero(n, n);
        radau_p1_.assign(static_cast<std::size_t>(n), 0);
        radau_p2_.assign(static_cast<std::size_t>(n), 0);
        radau_factored_version_ = -1;
        for (int i = 0; i < 3; ++i) {
            radau_z_[i] = VecX::Zero(n);
            radau_w_[i] = VecX::Zero(n);
            radau_f_[i] = VecX::Zero(n);
            radau_cont_[i] = VecX::Zero(n);
        }
        radau_end_ = VecX::Zero(n);
        radau_yn_ = VecX::Zero(n);
        radau_y_ = VecX::Zero(n);
        radau_f0_ = VecX::Zero(n);
        radau_r_ = VecX::Zero(n);
        radau_e_ = VecX::Zero(n);
        radau_rc_ = Eigen::VectorXcd::Zero(n);
        radau_cont_valid_ = false;
        radau_first_ = true;
    }

    // Starting values for a step of length `step`: the last step's
    // collocation polynomial, w(s) = s (c0 + (s - (c_2 - 1)) (c1 + (s - (c_1 -
    // 1)) c2)) in s = (t - t_n) / h_old, at the new stages' times, if nothing
    // has changed the state since it ended; zero otherwise.
    auto start = [&](Real step, bool continues) {
        for (int i = 0; i < 3; ++i) {
            if (!continues) {
                radau_z_[i].setZero();
                continue;
            }
            const Real s = rt.c[i] * step / radau_h_old_;
            radau_z_[i] = s * (radau_cont_[0] + (s - (rt.c[1] - 1.0)) * (radau_cont_[1] + (s - (rt.c[0] - 1.0)) * radau_cont_[2]));
        }
    };

    const Real t_end = time + dt;
    Real h = sc.adaptive && next_step_ > 0.0 ? std::min(next_step_, dt) : dt;
    bool after_rejection = false, have_f0 = false;
    int tried = 0;
    while (time < t_end) {
        if (++tried > sc.max_steps) detail::throw_too_many_steps("Radau IIA", sc.max_steps, dt, time);
        const Real remaining = t_end - time;
        const bool last = 1.01 * h >= remaining;
        const Real step = last ? remaining : h;

        radau_yn_.head(nq) = q;
        radau_yn_.segment(nq, nv) = v;
        radau_yn_.tail(nx) = x;
        const bool continues =
            radau_cont_valid_ && time == radau_end_time_ && (radau_yn_.array() == radau_end_.array()).all();
        start(step, continues);
        bool fresh = false;
        if (!stacked_jacobian_valid_) {
            stacked_jacobian(radau_yn_, time);
            fresh = true;
        }
        if (fresh) {
            radau_f0_ = stacked_f0_;
            have_f0 = true;
        }
        stacked_noise(radau_yn_);
        int iterations = 0;
        bool solved = false, singular = false;
        for (;;) {
            singular = !radau_factor(step);
            solved = !singular && radau_newton(step, iterations);
            if (solved || fresh) break;
            // J from an earlier state: build it here and solve again.
            stacked_jacobian(radau_yn_, time);
            fresh = true;
            radau_f0_ = stacked_f0_;
            have_f0 = true;
            stacked_noise(radau_yn_);
            iterations = 0;
            start(step, continues);
        }
        if (!solved) {
            if (!sc.adaptive) {
                detail::throw_unsolved("Radau IIA", step, time, !singular, implicit_control.max_iterations);
            }
            ++stats_.rejected;
            h = 0.5 * step;
            if (h < sc.min_step || time + h == time) {
                detail::throw_step_too_short("Radau IIA", h, time, sc.min_step, std::numeric_limits<Real>::infinity());
            }
            after_rejection = true;
            continue;
        }

        Real error = 0.0;
        if (sc.adaptive) {
            if (!have_f0) {
                stacked_f(radau_yn_, time, radau_f0_);
                have_f0 = true;
            }
            const Real atol = sc.absolute_tolerance, rtol = sc.relative_tolerance;
            auto measure = [&] {
                const Real sum = (radau_e_.array()
                                  / (atol + rtol * radau_yn_.array().abs().max((radau_yn_ + radau_z_[2]).array().abs())))
                                     .square()
                                     .sum();
                return n > 0 ? std::sqrt(sum / static_cast<Real>(n)) : 0.0;
            };
            radau_e_ = rt.dd[0] * radau_z_[0] + rt.dd[1] * radau_z_[1] + rt.dd[2] * radau_z_[2] + step * radau_f0_;
            lu_solve(radau_e1_, radau_p1_, radau_e_);
            error = measure();
            if (!(error <= 1.0) && std::isfinite(error) && (radau_first_ || after_rejection)) {
                radau_y_ = radau_yn_ + radau_e_;
                stacked_f(radau_y_, time, radau_r_);
                radau_e_ = rt.dd[0] * radau_z_[0] + rt.dd[1] * radau_z_[1] + rt.dd[2] * radau_z_[2] + step * radau_r_;
                lu_solve(radau_e1_, radau_p1_, radau_e_);
                error = measure();
            }
            if (!(error <= 1.0)) {
                ++stats_.rejected;
                h = step * (std::isfinite(error) ? std::max(0.2, 0.9 * std::pow(error, -0.25)) : 0.2);
                if (h < sc.min_step || time + h == time) {
                    detail::throw_step_too_short("Radau IIA", h, time, sc.min_step, error);
                }
                after_rejection = true;
                continue;
            }
        }

        // Accepted: the new state is y_n + Z_3. The collocation polynomial
        // through (0, -Z_3), (c_1 - 1, Z_1 - Z_3), (c_2 - 1, Z_2 - Z_3) and (0,
        // 0), in s = (t - t_{n+1}) / h, relative to the new state, in Newton's
        // form.
        q = radau_yn_.head(nq) + radau_z_[2].head(nq);
        v = radau_yn_.segment(nq, nv) + radau_z_[2].segment(nq, nv);
        x = radau_yn_.tail(nx) + radau_z_[2].tail(nx);
        normalize(model, q);
        const Real c1m1 = rt.c[0] - 1.0, c2m1 = rt.c[1] - 1.0, c1mc2 = rt.c[0] - rt.c[1];
        radau_cont_[0] = (radau_z_[1] - radau_z_[2]) / c2m1;
        radau_y_ = (radau_z_[0] - radau_z_[1]) / c1mc2;                       // a divided difference
        radau_cont_[2] = (radau_y_ - radau_z_[0] / rt.c[0]) / rt.c[1];      // another, for the moment
        radau_cont_[1] = (radau_y_ - radau_cont_[0]) / c1m1;
        radau_cont_[2] = radau_cont_[1] - radau_cont_[2];
        radau_h_old_ = step;
        time = last ? t_end : time + step;
        ++stats_.steps;
        stats_.step_size = step;
        stats_.error = error;
        if (sc.adaptive) {
            const Real grow = after_rejection ? 1.0 : 5.0;
            const Real proposal =
                step * (error > 0.0 ? std::min(grow, std::max(0.2, 0.9 * std::pow(error, -0.25))) : grow);
            h = last && step < h ? std::max(h, proposal) : proposal;
        }
        after_rejection = false;
        have_f0 = false;
        radau_first_ = false;
        stacked_jacobian_policy(3 * iterations);
        project_step();
        radau_end_.head(nq) = q;
        radau_end_.segment(nq, nv) = v;
        radau_end_.tail(nx) = x;
        radau_end_time_ = time;
        radau_cont_valid_ = true;
    }
    next_step_ = h;
}

} // namespace mbd::kernel
