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

/// P^T A P = L D L^T with diagonal pivoting, for a symmetric positive
/// semi-definite A, stopped at its numerical rank: at each step the largest
/// remaining diagonal is the pivot, and the factorization ends when it falls
/// below kRankTolerance of the first (as LAPACK's pstrf does). The rows left
/// are dependent on those factorized; solve() gives their components zero.
/// The constraint solver, the initial-condition assembly and statics count
/// rank by this one rule.
///
/// Stopping matters. Eliminating past the rank, as Eigen's LDLT does (it is
/// not rank-revealing), divides by pivots made of rounding error; the
/// quotients can be large enough to pass for pivots and to spoil the solve.
class RankRevealingLDLT {
public:
    /// The factorization stops at a pivot below this fraction of the first.
    static constexpr Real kRankTolerance = 1e-10;

    /// Storage for n x n matrices.
    explicit RankRevealingLDLT(Index n = 0);

    /// Factorize A, of which only the lower triangle is read. Allocates
    /// nothing when A has the size given at construction.
    void compute(const MatX& A);

    /// Pivots kept: the numerical rank of A.
    int rank() const { return rank_; }

    /// The row of A taken as the k-th pivot. The first rank() of them index
    /// a set of independent rows (and, A being symmetric, columns).
    Index pivot(Index k) const { return pivot_[static_cast<std::size_t>(k)]; }

    /// x with A x = r, for r in the range of A; the components of the rows
    /// not factorized are zero. x must not be r. Allocates nothing when x has
    /// the right size.
    void solve(const VecX& r, VecX& x) const;

private:
    MatX LD_;                     ///< L below the diagonal, D on it (pivoted order)
    std::vector<Index> pivot_;    ///< Row of A at each position
    mutable VecX work_;
    int rank_{0};
};

/// How a projection onto the constraints went.
struct ProjectionInfo {
    int iterations{0};             ///< Gauss-Newton steps taken
    Real initial_position_residual{0.0};   ///< |phi| before: the drift it corrected
    Real position_residual{0.0};   ///< |phi| at the end
    /// |J v - nu| before the velocities were moved, at the final q
    Real initial_velocity_residual{0.0};
    Real velocity_residual{0.0};   ///< |J v - nu| at the end
    bool converged{false};
};

class ConstraintSolver {
public:
    /// Equations whose pivot is below this fraction of the largest are
    /// treated as redundant.
    static constexpr Real kRankTolerance = RankRevealingLDLT::kRankTolerance;

    /// Sizes all working storage. `model` must outlive the solver.
    ConstraintSolver(const Model& model,
                     std::vector<std::shared_ptr<const ConstraintModel>> constraints);

    /// Number of constraint equations.
    int size() const { return m_; }

    /// Baumgarte stabilization rates [1/s]. Zero disables it.
    Real baumgarte_alpha{0.0};
    Real baumgarte_beta{0.0};

    /// When set, forward dynamics takes gamma as zero. At rest (v = 0) gamma
    /// holds only the prescribed accelerations of time-dependent constraints,
    /// so this freezes them where they are at t: statics has no inertia
    /// (static_equilibrium sets it while it evaluates at rest).
    bool quasi_static{false};

    /// phi, J, nu and gamma at (q, v, t), in phi(), J(), nu(), gamma().
    void evaluate(Data& data, const VecX& q, const VecX& v, Real t);

    /// Accelerations of the constrained system under the generalized forces
    /// tau, at (q, v, t). Returns v_dot; the multipliers are in lambda().
    const VecX& forward_dynamics(Data& data, const VecX& q, const VecX& v,
                                 const VecX& tau, Real t);

    /// The same, at the state of the last forward_kinematics(q, v, 0) into
    /// `data`, without a kinematics pass of its own: for a caller that needed
    /// the kinematics already (the simulator, for the forces).
    const VecX& forward_dynamics_from_kinematics(Data& data, const VecX& v,
                                                 const VecX& tau, Real t);

    /// The change dv with W dv = r + J^T mu and J dv = 0, J the constraint
    /// Jacobian of the last forward dynamics or evaluate(): r's motion
    /// restricted to what the constraints allow, weighed by W instead of M,
    /// and mu the multipliers that restrict it (for the linearly implicit
    /// step, task 4.5, W = M - h C - h^2 K). W must be symmetric positive
    /// definite; false, and dv unchanged, if its Cholesky factorization fails.
    /// Allocates nothing after the first call with W of this size.
    bool constrained_solve(const MatX& W, const VecX& r, VecX& dv);
    /// Move q onto phi(q, t) = 0, then v onto J v = nu, each by the change of
    /// least kinetic-energy norm (M-weighted). Gauss-Newton on q with the mass
    /// matrix of the starting point, each step halved until |phi| decreases.
    /// q and v are changed in place.
    ProjectionInfo project(Data& data, VecX& q, VecX& v, Real t,
                           Real tolerance = 1e-10, int max_iterations = 20);

    /// When set, project() weighs by the mass matrix that the last
    /// forward_dynamics factorized, if there was one, instead of computing it
    /// at q. Right after a time step that matrix belongs to a nearby state;
    /// any positive definite weight gives a valid projection, and only which
    /// point of the constraint manifold is chosen changes, to second order.
    bool reuse_mass_matrix{false};

    const VecX& phi() const { return phi_; }
    const MatX& J() const { return J_; }
    const VecX& nu() const { return nu_; }
    const VecX& gamma() const { return gamma_; }
    const VecX& v_dot() const { return v_dot_; }
    const VecX& lambda() const { return lambda_; }
    const SolveInfo& info() const { return info_; }

    /// The k-th of the info().rank independent equations of the last forward
    /// dynamics, k < info().rank: the order in which its rank-revealing
    /// factorization took them as pivots (task 4.7).
    Index independent_equation(int k) const { return ldlt_A_.pivot(k); }

    /// out += M^-1 J^T x, with M and J those of the last forward dynamics: the
    /// accelerations that extra constraint forces J^T x would add, for x with
    /// one entry per equation (task 4.7). Allocates nothing.
    void add_constraint_motion(const VecX& x, VecX& out);

    const std::vector<std::shared_ptr<const ConstraintModel>>& constraints() const
    {
        return constraints_;
    }

private:
    /// phi, J, nu and gamma from `data`, which holds forward_kinematics(q, v, 0).
    void calc_constraints(const Data& data, Real t);

    /// With llt_M_ holding M = L L^T: Z_ = L^-1 J^T and A_ = J M^-1 J^T =
    /// Z_^T Z_, factorized; sets info_.rank.
    void factorize_constraint_matrix();

    /// x with A x = r, dropping the pivots of redundant equations.
    void solve_constraint_matrix(const VecX& r, VecX& x);

    const Model& model_;
    std::vector<std::shared_ptr<const ConstraintModel>> constraints_;
    int m_{0};

    VecX phi_, nu_, gamma_, lambda_, rhs_m_, mu_;
    MatX J_, Z_, A_;
    VecX q_save_, v_dot_, v_dot_free_, rhs_v_, dv_, zero_v_, work_v_;
    Eigen::LLT<MatX> llt_M_;
    Eigen::LLT<MatX> llt_W_;   ///< constrained_solve's W
    RankRevealingLDLT ldlt_A_;
    bool mass_matrix_factorized_{false};
    SolveInfo info_;
};

} // namespace mbd::kernel
