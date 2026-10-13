#include "mbd/kernel/assembly.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "checks.hpp"
#include "labels.hpp"
#include "least_change.hpp"

namespace mbd::kernel {

namespace {

using detail::sci;

/// The held indices, sorted and without repeats; each must be a velocity.
std::vector<int> held_indices(std::vector<int> held, int nv)
{
    for (int k : held) {
        MBD_THROW_IF(k < 0 || k >= nv,
                     "MBD-K060: kernel::assemble: cannot hold velocity " + std::to_string(k)
                         + "; the model's velocities are 0 to " + std::to_string(nv - 1) + ".");
    }
    std::sort(held.begin(), held.end());
    held.erase(std::unique(held.begin(), held.end()), held.end());
    return held;
}

using detail::LeastChange;

} // namespace

AssemblyOptions& AssemblyOptions::hold_joint(const Model& model, int body, bool position, bool velocity)
{
    MBD_THROW_IF(body < 1 || body >= model.nbodies(),
                 "MBD-K060: AssemblyOptions::hold_joint: there is no jointed body "
                     + std::to_string(body) + "; the jointed bodies are 1 to "
                     + std::to_string(model.nbodies() - 1) + ".");
    const auto b = static_cast<std::size_t>(body);
    for (int k = model.idx_v[b]; k < model.idx_v[b] + model.nvs[b]; ++k) {
        if (position) hold_position.push_back(k);
        if (velocity) hold_velocity.push_back(k);
    }
    return *this;
}

AssemblyReport assemble(Simulator& sim, const AssemblyOptions& options)
{
    const Model& model = sim.system.model;
    checks::q("kernel::assemble", model, sim.q);
    checks::v("kernel::assemble", model, sim.v);
    const int nv = model.nv;
    const Real tol = options.tolerance;
    const Real t = sim.time;

    AssemblyReport r;
    r.time = t;
    r.held_positions = held_indices(options.hold_position, nv);
    r.held_velocities = held_indices(options.hold_velocity, nv);
    r.q_given = sim.q;
    r.v_given = sim.v;

    ConstraintSolver& solver = sim.solver();
    r.equations = solver.size();
    Data data(model);
    const VecX zero = VecX::Zero(nv);

    // --- Positions, first stage: Gauss-Newton onto phi = 0 ---------------------
    const VecX w_q = detail::mass_weights("kernel::assemble", model, data, r.q_given);
    LeastChange position(nv, r.held_positions);
    VecX q = r.q_given, step(nv), d(nv), target(nv), rhs;
    solver.evaluate(data, q, zero, t);
    r.position_residual_given = solver.phi().norm();
    Real residual = 0.0;
    r.projection_iterations = detail::project_positions(model, solver, data, position, w_q, q, t, tol,
                                                        options.max_iterations, residual);
    r.positions_assembled = residual <= tol;
    r.smallest_correction = r.positions_assembled && r.equations == 0;

    // --- Second stage: the least correction from the given positions ----------
    if (r.positions_assembled && r.equations > 0) {
        const VecX q_projected = q;
        Real last_step = std::numeric_limits<Real>::infinity();
        bool converged = false, diverging = false;
        while (!converged && !diverging && r.refinement_iterations < options.max_iterations) {
            // At q: d = difference(q_given, q) on the free velocities; the new
            // correction is the least with J d_new = J d - phi.
            difference(model, r.q_given, q, d);
            for (int k : r.held_positions) d(k) = 0.0;
            position.factorize(solver.J(), w_q);
            rhs = solver.J() * d - solver.phi();
            position.solve(rhs, target);
            step = target - d;
            integrate(model, q, step, 1.0, q);
            solver.evaluate(data, q, zero, t);
            ++r.refinement_iterations;
            const Real size = step.norm();
            converged = size <= tol && solver.phi().norm() <= tol;
            diverging = !converged && size >= last_step;
            last_step = size;
        }
        r.smallest_correction = converged;
        if (!converged) {
            // The first stage's positions satisfy the constraints all the same.
            q = q_projected;
            solver.evaluate(data, q, zero, t);
            r.warnings.push_back(
                "MBD-K064: Assembly: the smallest correction was not found ("
                + std::string(diverging ? "the steps grew" : "no convergence") + " after "
                + std::to_string(r.refinement_iterations) + " iterations, last step " + sci(last_step)
                + "). The positions satisfy the constraints but may be farther from the given ones "
                  "than necessary.");
        }
    }
    r.position_residual = solver.phi().norm();
    r.q = q;
    difference(model, r.q_given, q, r.position_change);

    // --- Velocities: one linear solve at the assembled positions -------------
    const VecX w_v = detail::mass_weights("kernel::assemble", model, data, q);
    solver.evaluate(data, q, r.v_given, t);
    const MatX& J = solver.J();
    rhs = solver.nu() - J * r.v_given;
    r.velocity_residual_given = rhs.norm();
    LeastChange velocity(nv, r.held_velocities);
    velocity.factorize(J, w_v);
    velocity.solve(rhs, step);
    r.v = r.v_given + step;
    r.velocity_residual = (J * r.v - solver.nu()).norm();
    r.velocities_assembled = r.velocity_residual <= tol;

    // --- Counts, with and without the held coordinates, at one place ---------
    LeastChange all(nv, {});
    all.factorize(J, w_v);
    r.rank = all.rank();
    r.degrees_of_freedom = nv - r.rank;
    position.factorize(J, w_v);
    r.position_hold_rank = position.rank();
    r.velocity_hold_rank = velocity.rank();

    if (!r.positions_assembled) {
        r.warnings.push_back("MBD-K062: Assembly: the positions were not assembled: |phi| = "
                             + sci(r.position_residual) + " after "
                             + std::to_string(r.projection_iterations) + " iterations (tolerance "
                             + sci(tol) + ").");
    }
    if (r.position_hold_rank < r.rank) {
        r.warnings.push_back(
            "MBD-K063: Assembly: the coordinates held in position are not independent of the "
            "constraints: with them held the constraint equations have rank "
            + std::to_string(r.position_hold_rank) + " instead of " + std::to_string(r.rank) + ", so "
            + std::to_string(r.rank - r.position_hold_rank)
            + " of them are already determined by the others. Hold fewer, or give them consistent values.");
    }
    if (!r.velocities_assembled) {
        r.warnings.push_back("MBD-K062: Assembly: the velocities were not assembled: |J v - nu| = "
                             + sci(r.velocity_residual) + " (tolerance " + sci(tol) + ").");
    }
    if (r.velocity_hold_rank < r.rank) {
        r.warnings.push_back(
            "MBD-K063: Assembly: the velocities held are not independent of the constraints: with "
            "them held the constraint equations have rank "
            + std::to_string(r.velocity_hold_rank) + " instead of " + std::to_string(r.rank) + ", so "
            + std::to_string(r.rank - r.velocity_hold_rank)
            + " of them are already determined by the others. Hold fewer, or give them consistent values.");
    }

    // --- Accelerations and forces at the assembled state ---------------------
    sim.q = r.q;
    sim.v = r.v;
    sim.refresh();
    r.loads = compute_loads(sim);
    r.acceleration_residual = (solver.J() * r.loads.v_dot - solver.gamma()).norm();
    sim.refresh();
    return r;
}

std::string AssemblyReport::summary(const Model& model) const
{
    std::ostringstream os;
    os.precision(6);
    const int nv = model.nv;
    const int free_left = nv - static_cast<int>(held_positions.size()) - position_hold_rank;
    os << "Assembly at t = " << time << " s: " << (ok() ? "assembled" : "NOT assembled") << ".\n"
       << "Constraints: " << equations << " equations of rank " << rank << "; velocities " << nv
       << "; degrees of freedom " << degrees_of_freedom << ".\n"
       << "Held: " << held_positions.size() << " in position, " << held_velocities.size()
       << " in velocity; degrees of freedom left " << free_left << ".\n"
       << "Positions: |phi| " << sci(position_residual_given) << " -> " << sci(position_residual)
       << ", " << projection_iterations << " projection and " << refinement_iterations
       << " refinement iterations" << (smallest_correction ? ", smallest correction" : "") << ".\n"
       << "Velocities: |J v - nu| " << sci(velocity_residual_given) << " -> "
       << sci(velocity_residual) << ".\n"
       << "Accelerations: |J v_dot - gamma| " << sci(acceleration_residual) << ".\n";

    auto held = [](const std::vector<int>& list, int k) {
        return std::binary_search(list.begin(), list.end(), k);
    };
    // One line per velocity, if this report is of this model.
    const bool of_model = q_given.size() == model.nq && q.size() == model.nq
                          && v_given.size() == nv && v.size() == nv && position_change.size() == nv;
    for (int k = 0; of_model && k < nv; ++k) {
        os << "  [" << k << "] " << detail::velocity_label(model, k) << ": position "
           << detail::position_text(model, k, q_given, q, position_change, held(held_positions, k))
           << "; velocity " << v_given(k);
        if (held(held_velocities, k)) os << " (held)";
        else os << " -> " << v(k) << " (change " << v(k) - v_given(k) << ")";
        os << '\n';
    }
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    return os.str();
}

} // namespace mbd::kernel
