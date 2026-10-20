#include "conflicts.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <Eigen/Cholesky>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "labels.hpp"

namespace mbd::kernel::detail {

namespace {

/// Whether the rows `rows` of J dq = -phi have no solution to `tolerance`:
/// with B = J_rows and A = B B^T, dq = -B^T mu solves them when A mu = phi;
/// RankRevealingLDLT gives such a mu when phi is in A's range, and leaves
/// phi - A mu clear of zero when it is not.
bool inconsistent(const MatX& J, const VecX& phi, const std::vector<int>& rows, Real tolerance)
{
    const Index n = static_cast<Index>(rows.size());
    if (n == 0) return false;
    MatX B(n, J.cols());
    VecX r(n);
    for (Index i = 0; i < n; ++i) {
        B.row(i) = J.row(rows[static_cast<std::size_t>(i)]);
        r(i) = phi(rows[static_cast<std::size_t>(i)]);
    }
    const MatX A = B * B.transpose();
    RankRevealingLDLT ldlt(n);
    ldlt.compute(A);
    VecX mu(n);
    ldlt.solve(r, mu);
    return (r - A * mu).norm() > tolerance;
}

} // namespace

std::vector<int> conflicting_equations(const MatX& J, const VecX& phi, Real tolerance)
{
    std::vector<int> rows;
    for (Index i = 0; i < phi.size(); ++i) rows.push_back(static_cast<int>(i));
    if (!inconsistent(J, phi, rows, tolerance)) return {};
    // The deletion filter: leave each equation out for good if the rest
    // still have no solution.
    for (Index i = 0; i < phi.size(); ++i) {
        std::vector<int> without;
        for (int r : rows) {
            if (r != static_cast<int>(i)) without.push_back(r);
        }
        if (without.size() < rows.size() && inconsistent(J, phi, without, tolerance)) rows = without;
    }
    return rows;
}

std::vector<int> conflicting_equations(const Model& model, ConstraintSolver& solver, Data& data, const VecX& q_from,
                                       Real t, const std::vector<int>& held, Real tolerance)
{
    // Levenberg-Marquardt on |phi|^2 / 2: (J^T J + lambda D) dq = -J^T phi,
    // D the diagonal of J^T J (Marquardt's scaling) plus a floor, lambda
    // lowered after a step that reduces |phi| and raised after one that does
    // not. It stops where the gradient J^T phi has vanished against
    // |J| |phi|, or after 200 tries.
    const VecX zero = VecX::Zero(model.nv);
    VecX q = q_from, trial, dq;
    auto evaluate = [&](const VecX& at, MatX& J, VecX& phi) {
        solver.evaluate(data, at, zero, t);
        J = solver.J();
        for (int k : held) J.col(k).setZero();
        phi = solver.phi();
    };
    MatX J, J_trial;
    VecX phi, phi_trial;
    evaluate(q, J, phi);
    Real lambda = 1e-3;
    for (int iteration = 0; iteration < 200; ++iteration) {
        const VecX g = J.transpose() * phi;
        if (g.norm() <= 1e-14 * (1.0 + J.norm() * phi.norm())) break;
        MatX H = J.transpose() * J;
        const VecX d = H.diagonal().cwiseMax(1e-12 * (1.0 + H.diagonal().maxCoeff()));
        H.diagonal() += lambda * d;
        dq = H.ldlt().solve(-g);
        integrate(model, q, dq, 1.0, trial);
        evaluate(trial, J_trial, phi_trial);
        if (phi_trial.squaredNorm() < phi.squaredNorm()) {
            q = trial;
            J = J_trial;
            phi = phi_trial;
            lambda = std::max(1e-12, lambda / 3.0);
        } else {
            lambda *= 4.0;
            if (lambda > 1e12) break;
        }
    }
    return conflicting_equations(J, phi, tolerance);
}

std::string equation_label(const Model& model, const std::vector<std::shared_ptr<const ConstraintModel>>& constraints,
                           int row)
{
    int first = 0;
    for (std::size_t k = 0; k < constraints.size(); ++k) {
        const ConstraintModel& c = *constraints[k];
        if (row >= first + c.size()) {
            first += c.size();
            continue;
        }
        const int j = row - first;   // the equation within the constraint
        std::string s = "constraint " + std::to_string(k) + " (" + c.name() + "), its equation "
                        + std::to_string(j + 1) + " of " + std::to_string(c.size());
        if (const auto* set = dynamic_cast<const ConstraintSet*>(&c)) {
            int part_first = 0;
            for (const auto& part : set->parts()) {
                if (j < part_first + part->size()) {
                    s += std::string(" (") + part->name() + ")";
                    break;
                }
                part_first += part->size();
            }
        }
        std::vector<int> seen;
        std::string on;
        for (int b : c.bodies()) {
            bool repeated = false;
            for (int s2 : seen) repeated = repeated || s2 == b;
            if (repeated) continue;
            seen.push_back(b);
            on += (on.empty() ? "" : " and ") + (b == 0 ? std::string("body 0 (ground)") : body_label(model, b));
        }
        return s + ", on " + on;
    }
    return "equation " + std::to_string(row);
}

std::string conflict_text(const Model& model, const std::vector<std::shared_ptr<const ConstraintModel>>& constraints,
                          const std::vector<int>& rows)
{
    if (rows.empty()) {
        return "The linearised constraint equations can all hold there: no equations contradict each other; "
               "the iterations ran out, or the positions are far from any that meet the constraints.";
    }
    std::string s = "The equations that cannot all hold together: ";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        s += (i > 0 ? "; " : "") + equation_label(model, constraints, rows[i]);
    }
    return s + ".";
}

} // namespace mbd::kernel::detail
