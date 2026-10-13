#include "mbd/kernel/assembly.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "checks.hpp"

namespace mbd::kernel {

namespace {

std::string sci(Real x)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.2e", x);
    return text;
}

/// "body 2 (coupler)", or "body 2" when it has no name.
std::string body_label(const Model& model, int body)
{
    const std::string& name = model.name[static_cast<std::size_t>(body)];
    return "body " + std::to_string(body) + (name.empty() ? "" : " (" + name + ")");
}

/// The body whose joint has velocity index k.
int body_of_velocity(const Model& model, int k)
{
    for (int i = 1; i < model.nbodies(); ++i) {
        if (k >= model.idx_v[i] && k < model.idx_v[i] + model.nvs[i]) return i;
    }
    return 0;
}

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

/// The weights of the corrections: the diagonal of the mass matrix at q.
VecX weights(const Model& model, Data& data, const VecX& q)
{
    const VecX w = crba(model, data, q).diagonal();
    for (int k = 0; k < model.nv; ++k) {
        MBD_THROW_IF(!(w(k) > 0.0),
                     "MBD-K061: kernel::assemble: velocity " + std::to_string(k) + " ("
                         + body_label(model, body_of_velocity(model, k))
                         + ") moves no mass, so its correction cannot be weighed (mass matrix entry "
                         + sci(w(k)) + "). validate() names such joints.");
    }
    return w;
}

/// The least changes x of the velocities not held, in the norm
/// sqrt(x^T W x) with W diagonal. With B = J_f W_f^-1/2 and A = B B^T,
/// x_f = W_f^-1/2 B^T mu with A mu = r solves J_f x_f = r: B^T mu lies in the
/// range of B^T, so it is the least solution, whichever mu solves A mu = r.
/// RankRevealingLDLT gives mu and the rank of J_f by the constraint solver's
/// own rule; when r is out of reach (held coordinates that over-determine the
/// constraints), it drops the dependent equations.
class LeastChange {
public:
    LeastChange(int nv, const std::vector<int>& held)
    {
        std::vector<bool> is_held(static_cast<std::size_t>(nv), false);
        for (int k : held) is_held[static_cast<std::size_t>(k)] = true;
        for (int k = 0; k < nv; ++k) {
            if (!is_held[static_cast<std::size_t>(k)]) free_.push_back(k);
        }
        scale_.resize(static_cast<Index>(free_.size()));
    }

    void factorize(const MatX& J, const VecX& w)
    {
        const Index nf = static_cast<Index>(free_.size());
        B_.resize(J.rows(), nf);
        for (Index c = 0; c < nf; ++c) {
            const int k = free_[static_cast<std::size_t>(c)];
            scale_(c) = 1.0 / std::sqrt(w(k));
            B_.col(c) = scale_(c) * J.col(k);
        }
        empty_ = B_.rows() == 0 || B_.cols() == 0;
        if (empty_) return;
        A_.noalias() = B_ * B_.transpose();
        ldlt_.compute(A_);
    }

    /// x (every velocity): the least change with J_f x_f = r; zero where held.
    void solve(const VecX& r, VecX& x)
    {
        x.setZero();
        if (empty_) return;
        ldlt_.solve(r, mu_);
        y_.noalias() = B_.transpose() * mu_;
        for (std::size_t c = 0; c < free_.size(); ++c) {
            x(free_[c]) = scale_(static_cast<Index>(c)) * y_(static_cast<Index>(c));
        }
    }

    int rank() const { return empty_ ? 0 : ldlt_.rank(); }

private:
    std::vector<int> free_;
    VecX scale_, mu_, y_;
    MatX B_, A_;
    RankRevealingLDLT ldlt_;
    bool empty_{true};
};

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
    const VecX w_q = weights(model, data, r.q_given);
    LeastChange position(nv, r.held_positions);
    VecX q = r.q_given, q_save, step(nv), d(nv), target(nv), rhs;
    solver.evaluate(data, q, zero, t);
    r.position_residual_given = solver.phi().norm();
    Real residual = r.position_residual_given;
    while (residual > tol && r.projection_iterations < options.max_iterations) {
        position.factorize(solver.J(), w_q);
        position.solve(-solver.phi(), step);
        q_save = q;
        const Real before = residual;
        Real fraction = 1.0;
        for (int halving = 0; halving < 12; ++halving) {
            integrate(model, q_save, step, fraction, q);
            solver.evaluate(data, q, zero, t);
            residual = solver.phi().norm();
            if (residual < before) break;
            fraction *= 0.5;
        }
        ++r.projection_iterations;
    }
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
    const VecX w_v = weights(model, data, q);
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
        const int b = body_of_velocity(model, k);
        const auto i = static_cast<std::size_t>(b);
        const int c = k - model.idx_v[i];
        // A joint whose coordinates are its velocities' integrals shows them;
        // a spherical or free joint shows its change only.
        const bool scalar = model.nqs[i] == model.nvs[i];
        os << "  [" << k << "] " << body_label(model, b) << ", " << model.joint[i]->name();
        if (model.nvs[i] > 1) os << " " << c;
        os << ": position ";
        if (scalar) {
            const int iq = model.idx_q[i] + c;
            os << q_given(iq);
            if (held(held_positions, k)) os << " (held)";
            else os << " -> " << q(iq) << " (change " << position_change(k) << ")";
        } else if (held(held_positions, k)) {
            os << "held";
        } else {
            os << "change " << position_change(k);
        }
        os << "; velocity " << v_given(k);
        if (held(held_velocities, k)) os << " (held)";
        else os << " -> " << v(k) << " (change " << v(k) - v_given(k) << ")";
        os << '\n';
    }
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    return os.str();
}

} // namespace mbd::kernel
