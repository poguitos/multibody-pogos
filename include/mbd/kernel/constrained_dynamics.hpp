#pragma once

// Constrained dynamics on the kinematics kernel (plan task 2.6): the one
// place that forms and solves the equations of a tree closed by constraints.
//
//   M(q) v_dot + b(q, v) = tau + J^T lambda
//   J v_dot              = gamma - 2 alpha (J v - nu) - beta^2 phi
//
// b holds the velocity-product and gravity terms (rnea at zero acceleration),
// J^T lambda the constraint forces. alpha and beta are Baumgarte's
// stabilization rates; zero by default, so the drift of phi is removed by
// projection instead.
//
// The equations are solved by the range-space method: with Y = M^-1 J^T,
// lambda solves (J Y) lambda = gamma - J M^-1 (tau - b). J Y is factorized as
// L D L^T with diagonal pivoting, which reveals its rank. When constraints are
// redundant (J Y singular), the pivots below kRankTolerance are dropped and
// the multipliers of the equations they belong to are zero; the constraint
// forces J^T lambda and the accelerations are unique all the same, and info()
// reports the rank.
//
// A solver holds working storage sized for one model and constraint set, so
// it does not allocate after construction. Like Data, it belongs to one
// thread; the Model and the constraints may be shared.

#include <memory>
#include <vector>

#include <Eigen/Cholesky>

#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel {

/// How the last solve went.
struct SolveInfo {
    int equations{0};  ///< Constraint equations
    int rank{0};       ///< Independent equations
    bool redundant() const { return rank < equations; }
};

/// How a projection onto the constraints went.
struct ProjectionInfo {
    int iterations{0};             ///< Gauss-Newton steps taken
    Real position_residual{0.0};   ///< |phi| at the end
    Real velocity_residual{0.0};   ///< |J v - nu| at the end
    bool converged{false};
};

class ConstraintSolver {
public:
    /// Equations whose pivot is below this fraction of the largest are
    /// treated as redundant.
    static constexpr Real kRankTolerance = 1e-10;

    /// Sizes all working storage. `model` must outlive the solver.
    ConstraintSolver(const Model& model,
                     std::vector<std::shared_ptr<const ConstraintModel>> constraints);

    /// Number of constraint equations.
    int size() const { return m_; }

    /// Baumgarte stabilization rates [1/s]. Zero disables it.
    Real baumgarte_alpha{0.0};
    Real baumgarte_beta{0.0};

    /// phi, J, nu and gamma at (q, v, t), in phi(), J(), nu(), gamma().
    void evaluate(Data& data, const VecX& q, const VecX& v, Real t);

    /// Accelerations of the constrained system under the generalized forces
    /// tau, at (q, v, t). Returns v_dot; the multipliers are in lambda().
    const VecX& forward_dynamics(Data& data, const VecX& q, const VecX& v,
                                 const VecX& tau, Real t);

    /// Move q onto phi(q, t) = 0, then v onto J v = nu, each by the change of
    /// least kinetic-energy norm (M-weighted). Gauss-Newton on q with the mass
    /// matrix of the starting point. q and v are changed in place.
    ProjectionInfo project(Data& data, VecX& q, VecX& v, Real t,
                           Real tolerance = 1e-10, int max_iterations = 20);

    const VecX& phi() const { return phi_; }
    const MatX& J() const { return J_; }
    const VecX& nu() const { return nu_; }
    const VecX& gamma() const { return gamma_; }
    const VecX& v_dot() const { return v_dot_; }
    const VecX& lambda() const { return lambda_; }
    const SolveInfo& info() const { return info_; }

    const std::vector<std::shared_ptr<const ConstraintModel>>& constraints() const
    {
        return constraints_;
    }

private:
    /// With llt_M_ holding M: Y_ = M^-1 J^T, A_ = J Y_, factorized; sets
    /// info_.rank.
    void factorize_constraint_matrix();

    /// x with A x = r, dropping the pivots of redundant equations.
    void solve_constraint_matrix(const VecX& r, VecX& x);

    const Model& model_;
    std::vector<std::shared_ptr<const ConstraintModel>> constraints_;
    int m_{0};

    VecX phi_, nu_, gamma_, lambda_, rhs_m_, mu_;
    MatX J_, Y_, A_;
    VecX v_dot_, v_dot_free_, rhs_v_, dv_, zero_v_;
    Eigen::LLT<MatX> llt_M_;
    Eigen::LDLT<MatX> ldlt_A_;
    Real pivot_cut_{0.0};
    SolveInfo info_;
};

} // namespace mbd::kernel
