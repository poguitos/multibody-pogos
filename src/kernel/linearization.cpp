#include "mbd/kernel/linearization.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "checks.hpp"
#include "labels.hpp"
#include "least_change.hpp"

namespace mbd::kernel {

namespace {

using detail::sci;

// The operating point must satisfy the constraints to this (the tolerance of
// assemble() and of the simulator's projection); the perturbed positions are
// projected far below the differences' step.
constexpr Real kOnConstraints = 1e-10;
constexpr Real kProjectionTolerance = 1e-13;
constexpr int kProjectionIterations = 50;
// Accelerations at rest above this mean the point is not an equilibrium
// (statics' default tolerance).
constexpr Real kEquilibrium = 1e-6;

Real largest(const VecX& a)
{
    return a.size() > 0 ? a.cwiseAbs().maxCoeff() : 0.0;
}

} // namespace

LinearModel linearize(Simulator& sim, const LinearizationOptions& options)
{
    const Model& model = sim.system.model;
    checks::q("kernel::linearize", model, sim.q);
    checks::v("kernel::linearize", model, sim.v);
    const int nv = model.nv;
    const Real t = sim.time;
    const Real h = options.perturbation;
    ConstraintSolver& solver = sim.solver();
    Data data(model);
    const VecX w = detail::mass_weights("kernel::linearize", model, data, sim.q);
    detail::LeastChange change(nv, {});

    LinearModel r;
    r.time = t;

    // The operating point on the constraints, positions then velocities.
    auto onto_constraints = [&](VecX& q, VecX& v) {
        Real residual = 0.0;
        detail::project_positions(model, solver, data, change, w, q, t, kProjectionTolerance,
                                  kProjectionIterations, residual);
        solver.evaluate(data, q, v, t);
        change.factorize(solver.J(), w);
        VecX dv;
        change.solve(solver.nu() - solver.J() * v, dv);
        v += dv;
    };
    VecX q0 = sim.q, v0 = sim.v;
    solver.evaluate(data, q0, v0, t);
    const Real phi_given = solver.phi().norm();
    const Real nu_given = (solver.J() * v0 - solver.nu()).norm();
    onto_constraints(q0, v0);
    if (phi_given > kOnConstraints || nu_given > kOnConstraints) {
        r.warnings.push_back("MBD-K080: Linearisation: the operating point did not satisfy the constraints "
                             "(|phi| = " + sci(phi_given) + ", |J v - nu| = " + sci(nu_given)
                             + "); it was linearised at the nearest point that does.");
    }
    r.q = q0;
    r.v = v0;

    // N: independent columns of the W-orthogonal projector P onto the
    // allowed motions, P e_k = e_k - (the least change x with J x = J e_k),
    // picked by the pivots of S P S^-1, a symmetric projector.
    solver.evaluate(data, q0, v0, t);
    change.factorize(solver.J(), w);
    const VecX s = w.cwiseSqrt();
    MatX P(nv, nv);
    VecX x;
    for (int k = 0; k < nv; ++k) {
        change.solve(solver.J().col(k), x);
        P.col(k) = -x;
        P(k, k) += 1.0;
    }
    RankRevealingLDLT pick(nv);
    pick.compute(s.asDiagonal() * P * s.cwiseInverse().asDiagonal());
    const int n = pick.rank();
    r.degrees_of_freedom = n;
    r.basis.resize(nv, n);
    for (int j = 0; j < n; ++j) r.basis.col(j) = P.col(pick.pivot(j));
    const MatX& N = r.basis;

    // The derivatives of the accelerations along each allowed motion.
    MatX Aq(nv, n), Av(nv, n);
    VecX qp, qm, vp, vm;
    for (int j = 0; j < n; ++j) {
        integrate(model, q0, N.col(j), h, qp);
        integrate(model, q0, N.col(j), -h, qm);
        vp = v0;
        vm = v0;
        onto_constraints(qp, vp);
        onto_constraints(qm, vm);
        VecX ap = sim.acceleration(qp, vp, t);
        Aq.col(j) = (ap - sim.acceleration(qm, vm, t)) / (2.0 * h);
        // N's columns satisfy J n = 0, so v0 +- h n keeps J v = nu.
        vp = v0 + h * N.col(j);
        vm = v0 - h * N.col(j);
        ap = sim.acceleration(q0, vp, t);
        Av.col(j) = (ap - sim.acceleration(q0, vm, t)) / (2.0 * h);
    }

    // Reduced by the W-weighted left inverse (N^T W N)^-1 N^T W.
    const MatX NtW = N.transpose() * w.asDiagonal();
    const Eigen::LLT<MatX> gram(NtW * N);
    r.A = MatX::Zero(2 * n, 2 * n);
    r.A.topRightCorner(n, n).setIdentity();
    r.A.bottomLeftCorner(n, n) = gram.solve(NtW * Aq);
    r.A.bottomRightCorner(n, n) = gram.solve(NtW * Av);

    const VecX a0 = sim.acceleration(q0, v0, t);
    r.residual = largest(a0);
    if (v0.norm() == 0.0 && r.residual > kEquilibrium) {
        r.warnings.push_back("MBD-K081: Linearisation: the operating point is at rest but not in equilibrium "
                             "(largest acceleration " + sci(r.residual)
                             + "): the model describes motion about a point the system leaves. "
                               "static_equilibrium() finds one.");
    }

    // The modes.
    if (n > 0) {
        const Eigen::EigenSolver<MatX> eigen(r.A, true);
        const Eigen::VectorXcd lambda = eigen.eigenvalues();
        const Eigen::MatrixXcd vectors = eigen.eigenvectors();
        const Eigen::MatrixXcd Nc = N.cast<std::complex<Real>>();
        const Eigen::MatrixXcd Mc = crba(model, data, q0).cast<std::complex<Real>>();
        for (Index i = 0; i < lambda.size(); ++i) {
            if (lambda(i).imag() < 0.0) continue;   // the twin of a listed pair
            Mode mode;
            mode.eigenvalue = lambda(i);
            mode.natural_frequency = std::abs(lambda(i));
            mode.damping_ratio = mode.natural_frequency > 0.0 ? -lambda(i).real() / mode.natural_frequency : 0.0;
            mode.shape = Nc * vectors.col(i).head(n);
            Index big = 0;
            mode.shape.cwiseAbs().maxCoeff(&big);
            if (std::abs(mode.shape(big)) > 0.0) mode.shape /= mode.shape(big);
            // The modal energy distribution: conj(phi_k) (M phi)_k over phi^H M phi.
            const Eigen::VectorXcd Mphi = Mc * mode.shape;
            const Real energy = mode.shape.dot(Mphi).real();   // dot conjugates the first
            mode.energy_fraction = VecX::Zero(nv);
            if (energy > 0.0) {
                for (Index k = 0; k < nv; ++k) {
                    mode.energy_fraction(k) = (std::conj(mode.shape(k)) * Mphi(k)).real() / energy;
                }
            }
            r.modes.push_back(std::move(mode));
        }
        std::sort(r.modes.begin(), r.modes.end(),
                  [](const Mode& a, const Mode& b) { return a.natural_frequency < b.natural_frequency; });
    }

    // Leave the simulator at the operating point.
    sim.q = q0;
    sim.v = v0;
    sim.refresh();
    return r;
}

std::string LinearModel::summary(const Model& model) const
{
    std::ostringstream os;
    os.precision(6);
    os << "Linear model at t = " << time << " s: " << degrees_of_freedom << " degrees of freedom, "
       << modes.size() << " modes; largest acceleration at the operating point " << sci(residual) << ".\n";
    for (std::size_t m = 0; m < modes.size(); ++m) {
        const Mode& mode = modes[m];
        os << "  mode " << m + 1 << ": lambda " << mode.eigenvalue.real() << (mode.eigenvalue.imag() < 0 ? " - " : " + ")
           << std::abs(mode.eigenvalue.imag()) << "i, " << mode.natural_frequency / (2.0 * pi) << " Hz, damping ratio "
           << mode.damping_ratio;
        if (mode.shape.size() == model.nv) {
            // The three largest components of the shape.
            std::vector<Index> order(static_cast<std::size_t>(model.nv));
            for (Index k = 0; k < model.nv; ++k) order[static_cast<std::size_t>(k)] = k;
            const auto& shape = mode.shape;
            std::partial_sort(order.begin(), order.begin() + std::min<Index>(3, model.nv), order.end(),
                              [&](Index a, Index b) { return std::abs(shape(a)) > std::abs(shape(b)); });
            os << "; shape:";
            for (Index i = 0; i < std::min<Index>(3, model.nv); ++i) {
                const Index k = order[static_cast<std::size_t>(i)];
                os << " [" << k << "] " << detail::velocity_label(model, static_cast<int>(k)) << " "
                   << std::abs(shape(k));
            }
            if (mode.energy_fraction.size() == model.nv) {
                // The coordinates holding most of its kinetic energy.
                const auto& e = mode.energy_fraction;
                std::partial_sort(order.begin(), order.begin() + std::min<Index>(3, model.nv), order.end(),
                                  [&](Index a, Index b) { return e(a) > e(b); });
                os << "; kinetic energy:";
                for (Index i = 0; i < std::min<Index>(3, model.nv); ++i) {
                    const Index k = order[static_cast<std::size_t>(i)];
                    os << " [" << k << "] " << detail::velocity_label(model, static_cast<int>(k)) << " "
                       << 100.0 * e(k) << " %";
                }
            }
        }
        os << '\n';
    }
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    return os.str();
}

} // namespace mbd::kernel
