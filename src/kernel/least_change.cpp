#include "least_change.hpp"

#include <string>

#include "mbd/kernel/algorithms.hpp"

#include "labels.hpp"

namespace mbd::kernel::detail {

VecX mass_weights(const char* function, const Model& model, Data& data, const VecX& q)
{
    const VecX w = crba(model, data, q).diagonal();
    for (int k = 0; k < model.nv; ++k) {
        MBD_THROW_IF(!(w(k) > 0.0),
                     "MBD-K061: " + std::string(function) + ": velocity " + std::to_string(k) + " ("
                         + body_label(model, body_of_velocity(model, k))
                         + ") moves no mass, so its correction cannot be weighed (mass matrix entry "
                         + sci(w(k)) + "). validate() names such joints.");
    }
    return w;
}

int project_positions(const Model& model, ConstraintSolver& solver, Data& data,
                      LeastChange& change, const VecX& w, VecX& q, Real t,
                      Real tolerance, int max_iterations, Real& residual)
{
    const VecX zero = VecX::Zero(model.nv);
    VecX step(model.nv), q_save;
    solver.evaluate(data, q, zero, t);
    residual = solver.phi().norm();
    int iterations = 0;
    while (residual > tolerance && iterations < max_iterations) {
        change.factorize(solver.J(), w);
        change.solve(-solver.phi(), step);
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
        ++iterations;
    }
    return iterations;
}

} // namespace mbd::kernel::detail
