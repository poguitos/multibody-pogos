#include "mbd/kernel/statics.hpp"

#include <cmath>
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

// The projections inside the differences go far below the differences' step,
// so that their error stays out of G.
constexpr Real kProjectionTolerance = 1e-13;
constexpr int kProjectionIterations = 50;
// The given positions must reach the constraints to this before statics
// starts (the tolerance of assemble() and of the simulator's projection).
constexpr Real kAssembledTolerance = 1e-10;

Real largest(const VecX& a)
{
    return a.size() > 0 ? a.cwiseAbs().maxCoeff() : 0.0;
}

/// Gives the solver back its quasi_static setting when it goes out of scope.
class FlagGuard {
public:
    explicit FlagGuard(ConstraintSolver& solver) : solver_(solver), saved_(solver.quasi_static) {}
    ~FlagGuard() { solver_.quasi_static = saved_; }
    FlagGuard(const FlagGuard&) = delete;
    FlagGuard& operator=(const FlagGuard&) = delete;

private:
    ConstraintSolver& solver_;
    bool saved_;
};

/// What one static analysis works with.
class Statics {
public:
    Statics(Simulator& sim, const StaticOptions& options)
        : sim_(sim)
        , model_(sim.system.model)
        , solver_(sim.solver())
        , data_(model_)
        , change_(model_.nv, {})
        , options_(options)
        , t_(sim.time)
        , zero_(VecX::Zero(model_.nv))
    {
        w_ = detail::mass_weights("kernel::static_equilibrium", model_, data_, sim.q);
        s_ = w_.cwiseSqrt();
        // Which velocities turn: those whose motion subspace has an angular
        // part (the joint frame's axes, so a hinge is a rotation wherever it
        // sits on the body).
        forward_kinematics(model_, data_, sim.q);
        rotation_.assign(static_cast<std::size_t>(model_.nv), false);
        for (int i = 1; i < model_.nbodies(); ++i) {
            const auto b = static_cast<std::size_t>(i);
            const Mat6X& S = data_.joint[b].S;
            for (Index c = 0; c < S.cols(); ++c) {
                rotation_[static_cast<std::size_t>(model_.idx_v[b] + c)] = S.col(c).head<3>().norm() > 0.0;
            }
        }
    }

    /// The step shortened so that no translation changes by more than
    /// max_translation and no rotation by more than max_rotation.
    void limit(VecX& step) const
    {
        Real factor = 1.0;
        for (Index k = 0; k < step.size(); ++k) {
            const Real cap = rotation_[static_cast<std::size_t>(k)] ? options_.max_rotation
                                                                    : options_.max_translation;
            if (std::abs(step(k)) * factor > cap) factor = cap / std::abs(step(k));
        }
        step *= factor;
    }

    /// q onto the constraints, in the weights; returns |phi| at the end, and
    /// leaves phi and J at q in the solver.
    Real project(VecX& q)
    {
        Real residual = 0.0;
        detail::project_positions(model_, solver_, data_, change_, w_, q, t_, kProjectionTolerance,
                                  kProjectionIterations, residual);
        return residual;
    }

    /// The degrees of freedom: the velocities less the rank of the
    /// constraints, at the configuration of the last projection.
    int degrees_of_freedom()
    {
        change_.factorize(solver_.J(), w_);
        return model_.nv - change_.rank();
    }

    /// The accelerations at rest at q, the drivers frozen.
    VecX rest(const VecX& q)
    {
        solver_.quasi_static = true;
        return sim_.acceleration(q, zero_, t_);
    }

    /// |S a|, which the line search decreases.
    Real merit(const VecX& a) const { return s_.cwiseProduct(a).norm(); }

    /// G = da/dq along the constraints, by central differences of projected
    /// configurations.
    void jacobian(const VecX& q, MatX& G)
    {
        const int nv = model_.nv;
        const Real h = options_.perturbation;
        G.resize(nv, nv);
        VecX e = VecX::Zero(nv), qp, qm;
        for (int k = 0; k < nv; ++k) {
            e(k) = 1.0;
            integrate(model_, q, e, h, qp);
            integrate(model_, q, e, -h, qm);
            e(k) = 0.0;
            project(qp);
            project(qm);
            G.col(k) = (rest(qp) - rest(qm)) / (2.0 * h);
        }
    }

    /// The least step, in the weights, with G step = -a; `rank` is the rank
    /// of G: the directions with stiffness.
    VecX newton_step(const MatX& G, const VecX& a, int& rank)
    {
        const MatX Gs = s_.asDiagonal() * G * s_.cwiseInverse().asDiagonal();
        const MatX A = Gs * Gs.transpose();
        RankRevealingLDLT ldlt(A.rows());
        ldlt.compute(A);
        rank = ldlt.rank();
        VecX y;
        ldlt.solve(-s_.cwiseProduct(a), y);
        return s_.cwiseInverse().cwiseProduct(Gs.transpose() * y);
    }

    /// Dynamic relaxation from q: a motion driven by the accelerations at
    /// rest, the velocities set to zero each time the kinetic energy has
    /// passed a maximum, until those accelerations are below the relaxation
    /// tolerance. The forces are taken at rest, as statics sees them: damping
    /// and tyre slip forces play no part in the equilibrium, and at low speed
    /// they are too stiff for an explicit step (a tyre's lateral force below
    /// its V_low acts as a damper of tens of kN s/m on an upright of a few
    /// tens of kg).
    void relax(VecX& q, StaticReport& r)
    {
        ++r.relaxation_runs;
        const Real dt = options_.relaxation_step;
        VecX v = VecX::Zero(model_.nv);
        VecX a = rest(q);
        Real kinetic_last = 0.0;
        while (r.relaxation_steps < options_.max_relaxation_steps) {
            v += dt * a;
            integrate(model_, q, v, dt, q);
            solver_.project(data_, q, v, t_);
            ++r.relaxation_steps;
            a = rest(q);
            if (largest(a) <= options_.relaxation_tolerance) break;
            forward_kinematics(model_, data_, q, v);
            const Real kinetic = kinetic_energy(model_, data_);
            const bool peak = kinetic < kinetic_last;
            if (peak) v.setZero();
            kinetic_last = peak ? 0.0 : kinetic;
        }
    }

private:
    Simulator& sim_;
    const Model& model_;
    ConstraintSolver& solver_;
    Data data_;
    detail::LeastChange change_;
    const StaticOptions& options_;
    Real t_;
    VecX zero_, w_, s_;
    std::vector<bool> rotation_;
};

} // namespace

StaticReport static_equilibrium(Simulator& sim, const StaticOptions& options)
{
    const Model& model = sim.system.model;
    checks::q("kernel::static_equilibrium", model, sim.q);
    const int nv = model.nv;
    StaticReport r;
    r.time = sim.time;
    r.q_given = sim.q;
    FlagGuard guard(sim.solver());
    Statics st(sim, options);

    VecX q = sim.q;
    const Real phi = st.project(q);
    r.degrees_of_freedom = st.degrees_of_freedom();
    if (phi > kAssembledTolerance) {
        r.warnings.push_back("MBD-K071: Static equilibrium: the given positions could not be brought onto "
                             "the constraints (|phi| = " + sci(phi)
                             + "). assemble() reports why.");
    } else if (nv > 0) {
        VecX a = st.rest(q), trial, a_trial;
        r.residual_given = largest(a);
        MatX G;
        int rank = 0;
        while (largest(a) > options.tolerance && r.iterations < options.max_iterations) {
            ++r.iterations;
            st.jacobian(q, G);
            VecX step = st.newton_step(G, a, rank);
            st.limit(step);
            const Real before = st.merit(a);
            bool accepted = false;
            Real fraction = 1.0;
            for (int halving = 0; halving < 12 && !accepted; ++halving, fraction *= 0.5) {
                integrate(model, q, step, fraction, trial);
                st.project(trial);
                a_trial = st.rest(trial);
                if (st.merit(a_trial) < before) {
                    q = trial;
                    a = a_trial;
                    accepted = true;
                }
            }
            r.history.push_back(largest(a));
            if (!accepted) {
                // No step reduces the accelerations: a direction with force
                // and no stiffness. Let the motion find the support.
                if (largest(a) <= options.relaxation_tolerance
                    || r.relaxation_steps >= options.max_relaxation_steps) {
                    break;
                }
                st.relax(q, r);
                a = st.rest(q);
            }
        }
        // The stiffness at the end: how many directions it holds.
        st.jacobian(q, G);
        st.newton_step(G, a, rank);
        r.supported = rank;
        r.neutral = r.degrees_of_freedom - r.supported;
    }

    // The result, at rest, with its loads.
    sim.q = q;
    sim.v.setZero();
    sim.refresh();
    sim.solver().quasi_static = true;
    r.loads = compute_loads(sim);
    sim.refresh();
    r.q = q;
    difference(model, r.q_given, q, r.position_change);
    r.residual = largest(r.loads.v_dot);
    r.converged = phi <= kAssembledTolerance && r.residual <= options.tolerance;

    if (!r.converged && phi <= kAssembledTolerance) {
        r.warnings.push_back("MBD-K070: Static equilibrium not found: largest acceleration " + sci(r.residual)
                             + " after " + std::to_string(r.iterations) + " Newton iterations and "
                             + std::to_string(r.relaxation_steps) + " relaxation steps (tolerance "
                             + sci(options.tolerance) + ").");
    }
    if (r.neutral > 0) {
        r.notes.push_back("MBD-K072: " + std::to_string(r.neutral) + " of the "
                          + std::to_string(r.degrees_of_freedom)
                          + " degrees of freedom have no stiffness at the equilibrium: it is neutral along "
                            "them, and the Newton steps leave them alone, so they keep their given values "
                            "but for effects of second order in the other changes.");
    }
    if (r.relaxation_steps > 0) {
        r.notes.push_back("MBD-K073: Dynamic relaxation took " + std::to_string(r.relaxation_steps)
                          + " steps in " + std::to_string(r.relaxation_runs)
                          + " runs, where Newton's method could not reduce the accelerations.");
    }
    return r;
}

std::string StaticReport::summary(const Model& model) const
{
    std::ostringstream os;
    os.precision(6);
    os << "Static equilibrium at t = " << time << " s: " << (converged ? "found" : "NOT found") << ".\n"
       << "Degrees of freedom " << degrees_of_freedom << ": " << supported << " with stiffness, "
       << neutral << " neutral.\n"
       << "Largest acceleration at rest: " << sci(residual_given) << " -> " << sci(residual) << "; "
       << iterations << " Newton iterations";
    if (!history.empty()) {
        os << " (";
        for (std::size_t i = 0; i < history.size(); ++i) os << (i ? ", " : "") << sci(history[i]);
        os << ")";
    }
    os << "; " << relaxation_steps << " relaxation steps in " << relaxation_runs << " runs.\n";
    const bool of_model = q_given.size() == model.nq && q.size() == model.nq
                          && position_change.size() == model.nv;
    for (int k = 0; of_model && k < model.nv; ++k) {
        os << "  [" << k << "] " << detail::velocity_label(model, k) << ": position "
           << detail::position_text(model, k, q_given, q, position_change, false) << '\n';
    }
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    for (const auto& m : notes) os << "Note: " << m << '\n';
    return os.str();
}

} // namespace mbd::kernel
