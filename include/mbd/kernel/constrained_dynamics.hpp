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

/// How one level of an assembly went (task 3.4, kernel/assembly.hpp).
struct AssemblyStepInfo {
    int iterations{0};       ///< Positions: Gauss-Newton steps, refinements included
    int refinements{0};      ///< Positions: steps towards the least correction
    Real last_change{0.0};   ///< Positions: how much the last refinement moved the correction
    Real residual{0.0};      ///< |phi|, |J v - nu| or |J a - gamma| at the end
    int rank{0};             ///< Rank of the constraints in the coordinates left free
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

    /// The same, at the state of the last forward_kinematics(q, v, 0) into
    /// `data`, without a kinematics pass of its own: for a caller that needed
    /// the kinematics already (the simulator, for the forces).
    const VecX& forward_dynamics_from_kinematics(Data& data, const VecX& v,
                                                 const VecX& tau, Real t);

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

    // --- Assembly (task 3.4) -------------------------------------------------
    //
    // Each moves its argument onto the constraints keeping the velocity
    // coordinates listed in `held` (indices 0 to nv - 1) exactly as given, and
    // changing the others by the least amount in the kinetic-energy metric:
    // the mass matrix with the held rows and columns removed, at the given q
    // for positions and at q for the rates. kernel::assemble() checks the
    // held lists and reports; these do the numerical work.

    /// phi(q, t) = 0. Gauss-Newton steps from q, each halved until |phi|
    /// decreases, reach the constraints; refinements then move along them
    /// until the correction q (-) q_given is the smallest, where it is a
    /// combination of the constraint directions (M d = J^T mu).
    AssemblyStepInfo assemble_positions(Data& data, VecX& q, Real t, const std::vector<int>& held,
                                        Real tolerance, int max_iterations);

    /// J v = nu at (q, t), in one solve: the problem is linear in v.
    AssemblyStepInfo assemble_velocities(Data& data, const VecX& q, VecX& v, Real t,
                                         const std::vector<int>& held, Real tolerance);

    /// J a = gamma at (q, v, t), in one solve.
    AssemblyStepInfo assemble_accelerations(Data& data, const VecX& q, const VecX& v, VecX& a,
                                            Real t, const std::vector<int>& held, Real tolerance);

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
    /// phi, J, nu and gamma from `data`, which holds forward_kinematics(q, v, 0).
    void calc_constraints(const Data& data, Real t);

    /// With llt_M_ holding M = L L^T: Z_ = L^-1 J^T and A_ = J M^-1 J^T =
    /// Z_^T Z_, factorized; sets info_.rank.
    void factorize_constraint_matrix();

    /// x with A x = r, dropping the pivots of redundant equations.
    void solve_constraint_matrix(const VecX& r, VecX& x);

    /// out += M^-1 J^T x = L^-T (Z_ x), without forming M^-1 J^T.
    void add_constraint_motion(const VecX& x, VecX& out);

    /// Gauss-Newton steps on phi(q, t) = 0 in the metric factorized in
    /// llt_M_, with the columns of `held` (if any) removed from J. Each step
    /// is halved until |phi| decreases; a step that cannot be made to
    /// decrease it ends the iteration at the best q. With
    /// `least_squares_fallback`, dependent equations whose plain step fails
    /// get a least-squares step first (assembly only; it allocates). Leaves
    /// phi, J and nu evaluated at the final q.
    ProjectionInfo gauss_newton(Data& data, VecX& q, Real t, Real tolerance, int max_iterations,
                                const std::vector<int>* held, bool least_squares_fallback);

    /// From q_save_, try q_save_ (+) (-step dv_) for step = 1, 1/2, ... until
    /// |phi| < before; q and residual hold the last point tried.
    bool halve_until_decrease(Data& data, VecX& q, Real t, Real before, Real& residual);

    /// dv_ = the least-squares Gauss-Newton step, from Z_ and phi_.
    void least_squares_step();

    /// The assembly metric: the mass matrix at q with the rows and columns
    /// of `held` replaced by those of the identity, factorized in llt_M_.
    void factorize_metric(Data& data, const VecX& q, const std::vector<int>& held);

    /// Zero the columns of J that belong to held coordinates.
    void remove_held_columns(const std::vector<int>& held);

    const Model& model_;
    std::vector<std::shared_ptr<const ConstraintModel>> constraints_;
    int m_{0};

    VecX phi_, nu_, gamma_, lambda_, rhs_m_, mu_;
    MatX J_, Z_, A_, M_metric_;
    VecX q_save_, q_given_, q_best_, d_;
    VecX v_dot_, v_dot_free_, rhs_v_, dv_, zero_v_, work_v_;
    Eigen::LLT<MatX> llt_M_;
    Eigen::LDLT<MatX> ldlt_A_;
    Real pivot_cut_{0.0};
    bool mass_matrix_factorized_{false};
    SolveInfo info_;
};

} // namespace mbd::kernel
