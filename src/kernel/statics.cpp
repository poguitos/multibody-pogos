#include "mbd/kernel/statics.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "held.hpp"
#include "labels.hpp"
#include "motions.hpp"

namespace mbd::kernel {

namespace {

// Tolerances of the projections inside statics: those of the simulator.
constexpr Real kProjectionTolerance = 1e-10;
constexpr int kProjectionIterations = 50;
// Stiffness eigenvalues within this fraction of the largest are taken as
// zero: the finite-difference noise of K is about eps |F| / h, some 1e-10 of
// the stiffest element's stiffness with h = 1e-6.
constexpr Real kStiffnessZero = 1e-8;

Real largest_abs(const VecX& x, int& at)
{
    Real m = 0.0;
    at = -1;
    for (Index k = 0; k < x.size(); ++k) {
        if (std::abs(x(k)) > m) {
            m = std::abs(x(k));
            at = static_cast<int>(k);
        }
    }
    return m;
}

// Everything statics evaluates at a configuration, with its own working
// storage; the simulator supplies the forces.
class Statics {
public:
    Statics(Simulator& sim, std::vector<int> held)
        : sim_(sim)
        , model_(sim.system.model)
        , data_(model_)
        , solver_(model_, sim.system.constraints)
        , held_(std::move(held))
        , t_(sim.time)
        , zero_(VecX::Zero(model_.nv))
    {
        is_held_.assign(static_cast<std::size_t>(model_.nv), false);
        for (int k : held_) is_held_[static_cast<std::size_t>(k)] = true;
    }

    /// f(q): the forces at rest, less gravity, without the constraint forces.
    const VecX& forces(const VecX& q)
    {
        f_ = sim_.applied_forces(q, zero_, t_);
        f_ -= rnea(model_, data_, q, zero_, zero_);
        return f_;
    }

    /// a(q), lambda(q) and merit(q) = a^T M a.
    void accelerations(const VecX& q)
    {
        forces(q);
        a_ = solver_.accelerations_at_rest(data_, q, f_, t_, held_);
        lambda_ = solver_.lambda();
        merit_ = a_.dot(data_.M * a_);
    }

    bool project(VecX& q)
    {
        return solver_.project_positions(data_, q, t_, held_, kProjectionTolerance,
                                         kProjectionIterations).converged;
    }

    /// F(q, lambda) = f(q) + J(q)^T lambda.
    const VecX& residual(const VecX& q, const VecX& lambda)
    {
        forces(q);
        if (solver_.size() > 0) {
            solver_.evaluate(data_, q, zero_, t_);
            F_ = f_;
            F_.noalias() += solver_.J().transpose() * lambda;
        } else {
            F_ = f_;
        }
        return F_;
    }

    /// K = dF/dq at fixed lambda, by central differences of step h, in the
    /// columns of the coordinates that are not held.
    void stiffness(const VecX& q, const VecX& lambda, Real h, MatX& K)
    {
        const int n = model_.nv;
        K.setZero(n, n);
        VecX e = zero_, q_plus(model_.nq), q_minus(model_.nq), F_plus(n);
        for (int k = 0; k < n; ++k) {
            if (is_held_[static_cast<std::size_t>(k)]) continue;
            e.setZero();
            e(k) = h;
            integrate(model_, q, e, 1.0, q_plus);
            integrate(model_, q, e, -1.0, q_minus);
            F_plus = residual(q_plus, lambda);
            K.col(k) = (F_plus - residual(q_minus, lambda)) / (2.0 * h);
        }
    }

    /// An orthonormal basis of the motions the constraints and the holds
    /// allow at q: J dq = 0 and dq_k = 0 for every held k.
    MatX allowed_motions(const VecX& q)
    {
        return motions::allowed(solver_, data_, q, t_, held_, model_.nv);
    }

    const VecX& a() const { return a_; }
    const VecX& lambda() const { return lambda_; }
    const VecX& f() const { return f_; }
    Real merit() const { return merit_; }
    const Model& model() const { return model_; }
    const MatX& mass_matrix() const { return data_.M; }

private:
    Simulator& sim_;
    const Model& model_;
    Data data_;
    ConstraintSolver solver_;
    std::vector<int> held_;
    std::vector<bool> is_held_;
    Real t_;
    VecX zero_, f_, F_, a_, lambda_;
    Real merit_{0.0};
};

// Eigenvalues of the symmetric part of the reduced stiffness -N^T K N: the
// count of zero ones (no restoring force) and of negative ones (unstable).
void classify_stiffness(const MatX& Kr, StaticsReport& r)
{
    if (Kr.rows() == 0) return;
    const MatX S = -0.5 * (Kr + Kr.transpose());
    const Eigen::SelfAdjointEigenSolver<MatX> eig(S);
    const VecX& ev = eig.eigenvalues();
    const Real scale = ev.cwiseAbs().maxCoeff();
    r.neutral_directions = 0;
    r.unstable_directions = 0;
    for (Index k = 0; k < ev.size(); ++k) {
        if (std::abs(ev(k)) <= kStiffnessZero * scale) {
            ++r.neutral_directions;
        } else if (ev(k) < 0.0) {
            ++r.unstable_directions;
        }
    }
}

} // namespace

StaticsReport static_equilibrium(Simulator& sim, const StaticsOptions& options)
{
    const Model& model = sim.system.model;
    Statics s(sim, held::checked(model, options.hold, "kernel::static_equilibrium", "coordinates"));
    StaticsReport r;

    VecX q = sim.q;
    auto finish = [&](StaticsReport& rep) -> StaticsReport {
        sim.q = q;
        sim.v.setZero();
        sim.refresh();
        return rep;
    };

    if (!s.project(q)) {
        r.errors.push_back("MBD-K071: Statics could not start: the configuration could not be brought "
                           "onto the constraints. Assemble it first (kernel::assemble), which "
                           "reports why.");
        return finish(r);
    }

    s.accelerations(q);
    VecX a = s.a();
    Real merit = s.merit();
    int at = -1;
    r.acceleration_before = largest_abs(a, at);
    r.acceleration = r.acceleration_before;
    r.largest_acceleration_at = at;
    r.history.push_back(r.acceleration);

    const Real h = options.stiffness_step;
    int relaxation_budget = options.dynamic_relaxation ? options.relaxation_max_steps : 0;
    MatX K, N, Kr;
    VecX dq, q_trial(model.nq);

    // Dynamic relaxation with kinetic damping, from q until |a|_inf falls to
    // `target`. False if the budget runs out, or if the motion blows up (the
    // step is too long for the stiffest motion), when q is left at the last
    // good point.
    auto relax = [&](Real target) {
        VecX v = VecX::Zero(model.nv);
        VecX q_good = q;
        Real ke_previous = 0.0;
        const Real dt = options.relaxation_step;
        while (relaxation_budget > 0) {
            s.accelerations(q);
            if (largest_abs(s.a(), at) <= target) return true;
            q_good = q;
            v += dt * s.a();
            integrate(model, q, v, dt, q);
            if (!q.allFinite() || !v.allFinite() || !s.project(q)) {
                q = q_good;
                return false;
            }
            const Real ke = 0.5 * v.dot(s.mass_matrix() * v);
            if (ke < ke_previous) {
                // The kinetic energy has passed a peak: the potential energy
                // is near a minimum along this swing. Start again from rest.
                v.setZero();
                ke_previous = 0.0;
            } else {
                ke_previous = ke;
            }
            --relaxation_budget;
            ++r.relaxation_steps;
        }
        return false;
    };

    while (r.acceleration > options.tolerance && r.iterations < options.max_iterations) {
        // The Newton step on the constraint surface.
        const VecX lambda = s.lambda();
        s.stiffness(q, lambda, h, K);
        N = s.allowed_motions(q);
        Kr = N.transpose() * K * N;
        const VecX rhs = -(N.transpose() * s.forces(q));
        // The threshold must be set before the decomposition: it decides the
        // rank the decomposition is built with, not only what rank() says.
        Eigen::CompleteOrthogonalDecomposition<MatX> cod;
        cod.setThreshold(kStiffnessZero);
        cod.compute(Kr);
        dq = N * cod.solve(rhs);
        ++r.iterations;

        // Halve it until the accelerations decrease.
        bool decreased = false;
        Real step = 1.0;
        for (int halving = 0; halving < 12 && !decreased; ++halving, step *= 0.5) {
            integrate(model, q, dq, step, q_trial);
            if (!q_trial.allFinite() || !s.project(q_trial)) continue;
            s.accelerations(q_trial);
            if (s.merit() < merit) {
                decreased = true;
                q = q_trial;
            }
        }

        if (!decreased) {
            // Nothing along the Newton direction helps: no stiffness where a
            // force acts (a body above the ground it will rest on), or a
            // kink in a force law. Let the forces move the system instead.
            if (relaxation_budget == 0) break;
            ++r.relaxations;
            const Real target = std::max(options.tolerance, 0.01 * r.acceleration);
            if (!relax(target)) {
                s.accelerations(q);
                r.acceleration = largest_abs(s.a(), r.largest_acceleration_at);
                break;
            }
            s.accelerations(q);
        }
        a = s.a();
        merit = s.merit();
        r.acceleration = largest_abs(a, r.largest_acceleration_at);
        r.history.push_back(r.acceleration);
    }
    r.converged = r.acceleration <= options.tolerance;

    // The stiffness at the point reached: its degrees of freedom, neutral
    // and unstable directions.
    s.accelerations(q);
    N = s.allowed_motions(q);
    r.degrees_of_freedom = static_cast<int>(N.cols());
    if (N.cols() > 0) {
        s.stiffness(q, s.lambda(), h, K);
        Kr = N.transpose() * K * N;
        classify_stiffness(Kr, r);
    }

    if (!r.converged) {
        r.errors.push_back("MBD-K070: No static equilibrium found: the largest acceleration at rest is "
                           + labels::number(r.acceleration) + " (tolerance "
                           + labels::number(options.tolerance) + "), in "
                           + coordinate_label(model, r.largest_acceleration_at) + ", after "
                           + std::to_string(r.iterations) + " Newton iterations and "
                           + std::to_string(r.relaxation_steps) + " relaxation steps.");
    }
    if (r.unstable_directions > 0) {
        r.warnings.push_back("MBD-K072: The equilibrium is unstable: the stiffness is negative in "
                             + std::to_string(r.unstable_directions)
                             + " direction(s), so the smallest disturbance moves the system away "
                               "from it (a pendulum balanced upright).");
    }
    if (r.neutral_directions > 0) {
        r.notes.push_back("MBD-K073: " + std::to_string(r.neutral_directions) + " of the "
                          + std::to_string(r.degrees_of_freedom)
                          + " degrees of freedom have no stiffness here: nothing pushes back "
                            "along them (a vehicle's position and heading on a flat road), "
                            "and statics left them as they were.");
    }
    if (r.relaxations > 0) {
        r.notes.push_back("MBD-K074: Newton could not reduce the accelerations "
                          + std::to_string(r.relaxations) + " time(s); dynamic relaxation took over for "
                          + std::to_string(r.relaxation_steps) + " steps.");
    }
    return finish(r);
}

std::string StaticsReport::summary(const Model& model) const
{
    std::ostringstream os;
    os << "statics: " << (converged ? "converged" : "NOT CONVERGED") << " in " << iterations
       << " Newton iterations and " << relaxation_steps << " relaxation steps; |a| "
       << labels::number(acceleration_before) << " -> " << labels::number(acceleration);
    if (largest_acceleration_at >= 0) os << ", largest in " << coordinate_label(model, largest_acceleration_at);
    os << "\nhistory:";
    for (Real x : history) os << ' ' << labels::number(x);
    os << "\ndegrees of freedom " << degrees_of_freedom << ": " << neutral_directions
       << " without stiffness, " << unstable_directions << " unstable\n";
    for (const auto& m : errors) os << "Error: " << m << '\n';
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    for (const auto& m : notes) os << "Note: " << m << '\n';
    return os.str();
}

} // namespace mbd::kernel
