#pragma once

// Which constraint equations conflict, when the positions cannot be brought
// onto the constraints (plan task 5.2, decision D44). Internal to the
// library: used by the initial-condition assembly (MBD-K062) and by the
// simulator's projection (MBD-K051).
//
// Where the constraints contradict each other, no positions meet them; the
// nearest come closest in least squares, a minimum of |phi|^2 / 2 where
// J^T phi = 0, so that phi is out of J's range and the linearised equations
// J dq = -phi have no solution. Gauss-Newton, as the projections take it,
// approaches that point slowly (each step aims at phi = 0, which is not
// there), so the diagnosis first goes to it by Levenberg-Marquardt, on a copy
// of the positions, the coordinates held fixed staying fixed. There a
// deletion filter (Chinneck and Dravnieks 1991)
// finds a smallest set of them that has none: starting from all, each
// equation in turn is left out for good if the others still have no
// solution. What remains cannot all hold, and every one of its equations is
// needed for that (an irreducible inconsistent subsystem of the linearised
// equations). Two distances from one point to two anchors farther apart than
// their sum give those two equations, and nothing else of the model. When the
// linearised equations do have a solution, the set is empty: the projection
// was short of iterations, or far from the constraints, not contradicted.

#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel::detail {

/// The rows of J dq = -phi of a smallest set with no solution to within
/// `tolerance` (in phi's units), in increasing order; empty if all of them
/// have one. For diagnostics: it allocates, and costs a factorization per
/// equation.
std::vector<int> conflicting_equations(const MatX& J, const VecX& phi, Real tolerance);

/// The same at the least-squares positions nearest q: Levenberg-Marquardt
/// on |phi(q, t)|^2 / 2 from q, the velocities in `held` kept, then the
/// deletion filter. q is not changed.
std::vector<int> conflicting_equations(const Model& model, ConstraintSolver& solver, Data& data, const VecX& q,
                                       Real t, const std::vector<int>& held, Real tolerance);

/// "constraint 1 (revolute closure), its equation 2 of 5 (point coincidence),
/// on body 3 (coupler) and body 0 (ground)": row `row` of the stacked
/// constraint equations.
std::string equation_label(const Model& model, const std::vector<std::shared_ptr<const ConstraintModel>>& constraints,
                           int row);

/// The sentence a message adds: "The equations that cannot all hold
/// together: <label>; <label>." or, for an empty set, that the linearised
/// equations can all hold.
std::string conflict_text(const Model& model, const std::vector<std::shared_ptr<const ConstraintModel>>& constraints,
                          const std::vector<int>& rows);

} // namespace mbd::kernel::detail
