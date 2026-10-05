#include "mbd/kernel/linearization.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "labels.hpp"
#include "motions.hpp"

namespace mbd::kernel {

namespace {

// The projections of the perturbed points start O(step^2) off the
// constraints; one Gauss-Newton step brings them to roundoff, below this.
constexpr Real kProjectionTolerance = 1e-14;
constexpr int kProjectionIterations = 20;
// The operating point counts as at rest in equilibrium below statics'
// default tolerance.
constexpr Real kEquilibriumTolerance = 1e-6;
// Eigenvalues within this fraction of the largest are taken as zero.
constexpr Real kZero = 1e-8;
// Time step of the central difference that gives dy/dt where the chart of
// the rotations is curved (a moving state with spherical or free joints).
constexpr Real kChartStep = 1e-4;

constexpr Real kTwoPi = 2.0 * 3.14159265358979323846;

bool has_rotation_joints(const Model& model)
{
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto& joint = model.joint[static_cast<std::size_t>(i)];
        if (!joint) continue;
        if (std::strcmp(joint->name(), "spherical") == 0 || std::strcmp(joint->name(), "free") == 0) {
            return true;
        }
    }
    return false;
}

class Linearizer {
public:
    Linearizer(Simulator& sim, const VecX& q0, const VecX& v0, MatX N)
        : sim_(sim)
        , model_(sim.system.model)
        , data_(model_)
        , solver_(model_, sim.system.constraints)
        , q0_(q0)
        , v0_(v0)
        , t_(sim.time)
        , N_(std::move(N))
        , rotations_(has_rotation_joints(model_))
        , d_(model_.nv)
    {
    }

    /// The point with reduced positions dy from the operating point: q0
    /// moved along N dy, projected onto the constraints, at the operating
    /// velocities projected onto J v = nu there.
    void at_positions(const VecX& dy, VecX& q, VecX& v)
    {
        integrate(model_, q0_, N_ * dy, 1.0, q);
        solver_.project_positions(data_, q, t_, none_, kProjectionTolerance, kProjectionIterations);
        v = v0_;
        solver_.assemble_velocities(data_, q, v, t_, none_, kProjectionTolerance);
    }

    /// The point with reduced velocities dz from the operating point.
    void at_velocities(const VecX& dz, VecX& q, VecX& v)
    {
        q = q0_;
        v = v0_ + N_ * dz;
        solver_.assemble_velocities(data_, q, v, t_, none_, kProjectionTolerance);
    }

    /// The reduced coordinates (y, z) a point actually has.
    void reduced(const VecX& q, const VecX& v, VecX& x)
    {
        const Index d = N_.cols();
        difference(model_, q0_, q, d_);
        x.resize(2 * d);
        x.head(d).noalias() = N_.transpose() * d_;
        x.tail(d).noalias() = N_.transpose() * v;
    }

    /// d/dt (y, z) at a point; reduced() must have been called for it.
    void derivative(const VecX& q, const VecX& v, VecX& f)
    {
        const Index d = N_.cols();
        f.resize(2 * d);
        const VecX& a = sim_.acceleration(q, v, t_);
        f.tail(d).noalias() = N_.transpose() * a;
        // dy/dt = N^T d/dt (q (-) q0). For scalar coordinates (-) is a
        // subtraction and this is N^T v; so it is for rotations at q0 or at
        // rest. Otherwise the curvature of the chart enters, and a central
        // difference in time gives it.
        const bool exact = !rotations_ || v.cwiseAbs().maxCoeff() == 0.0
                           || d_.cwiseAbs().maxCoeff() <= 1e-14;
        if (exact) {
            f.head(d).noalias() = N_.transpose() * v;
            return;
        }
        VecX q_plus(model_.nq), q_minus(model_.nq), d_plus(model_.nv), d_minus(model_.nv);
        integrate(model_, q, v, kChartStep, q_plus);
        integrate(model_, q, v, -kChartStep, q_minus);
        difference(model_, q0_, q_plus, d_plus);
        difference(model_, q0_, q_minus, d_minus);
        f.head(d).noalias() = N_.transpose() * (d_plus - d_minus) / (2.0 * kChartStep);
    }

    /// N^T M(q0) N.
    MatX reduced_mass()
    {
        crba(model_, data_, q0_);
        return N_.transpose() * data_.M * N_;
    }

private:
    Simulator& sim_;
    const Model& model_;
    Data data_;
    ConstraintSolver solver_;
    VecX q0_, v0_;
    Real t_;
    MatX N_;
    bool rotations_;
    VecX d_;
    const std::vector<int> none_;
};

template <class Vec>
int largest_entry(const Vec& x)
{
    int at = -1;
    Real m = 0.0;
    for (Index k = 0; k < x.size(); ++k) {
        if (std::abs(x(k)) > m) {
            m = std::abs(x(k));
            at = static_cast<int>(k);
        }
    }
    return at;
}

} // namespace

Linearization linearize(Simulator& sim, const LinearizationOptions& options)
{
    const Model& model = sim.system.model;
    const int nv = model.nv;
    const VecX q_saved = sim.q, v_saved = sim.v, tau_saved = sim.tau;
    Linearization L;

    // The operating point and the motions it allows.
    {
        Data data(model);
        ConstraintSolver solver(model, sim.system.constraints);
        L.N = motions::allowed(solver, data, sim.q, sim.time, {}, nv);
    }
    const Index d = L.N.cols();
    L.operating_acceleration = sim.acceleration(sim.q, sim.v, sim.time).cwiseAbs().maxCoeff();
    L.operating_velocity = nv > 0 ? sim.v.cwiseAbs().maxCoeff() : 0.0;
    if (L.operating_acceleration > kEquilibriumTolerance || L.operating_velocity > 0.0) {
        L.warnings.push_back("MBD-K080: The operating point is not at rest in equilibrium (largest "
                             "acceleration " + labels::number(L.operating_acceleration)
                             + ", largest velocity " + labels::number(L.operating_velocity)
                             + "): A describes the motion near it, but its eigenvalues are not "
                               "modes of vibration about it. Run static_equilibrium first.");
    }
    if (d == 0) {
        sim.refresh();
        return L;
    }

    Linearizer lin(sim, sim.q, sim.v, L.N);
    const Real h = options.step;
    MatX dX(2 * d, 2 * d), dF(2 * d, 2 * d);
    VecX q(model.nq), v(nv), x_plus, x_minus, f_plus, f_minus;
    for (Index j = 0; j < 2 * d; ++j) {
        for (const Real sign : {1.0, -1.0}) {
            VecX delta = VecX::Zero(d);
            delta(j < d ? j : j - d) = sign * h;
            if (j < d) {
                lin.at_positions(delta, q, v);
            } else {
                lin.at_velocities(delta, q, v);
            }
            VecX& x = sign > 0.0 ? x_plus : x_minus;
            VecX& f = sign > 0.0 ? f_plus : f_minus;
            lin.reduced(q, v, x);
            lin.derivative(q, v, f);
        }
        dX.col(j) = x_plus - x_minus;
        dF.col(j) = f_plus - f_minus;
    }
    // A dX = dF.
    L.A = dX.transpose().partialPivLu().solve(dF.transpose()).transpose();

    // Inputs: the accelerations are affine in tau, so a unit difference is
    // exact up to roundoff.
    if (options.inputs) {
        L.B = MatX::Zero(2 * d, nv);
        for (int k = 0; k < nv; ++k) {
            sim.tau = tau_saved;
            sim.tau(k) += 1.0;
            const VecX a_plus = sim.acceleration(q_saved, v_saved, sim.time);
            sim.tau(k) -= 2.0;
            const VecX a_minus = sim.acceleration(q_saved, v_saved, sim.time);
            L.B.col(k).tail(d) = 0.5 * L.N.transpose() * (a_plus - a_minus);
        }
        sim.tau = tau_saved;
    }

    // The second-order form.
    L.M = lin.reduced_mass();
    L.K = -L.M * L.A.block(d, 0, d, d);
    L.C = -L.M * L.A.block(d, d, d, d);

    // Modes from A.
    const Eigen::EigenSolver<MatX> es(L.A);
    const auto& lambda = es.eigenvalues();
    const auto& V = es.eigenvectors();
    const Real scale = std::max(lambda.cwiseAbs().maxCoeff(), Real(1e-300));
    int growing = 0;
    for (Index i = 0; i < lambda.size(); ++i) {
        const std::complex<Real> l = lambda(i);
        if (l.real() > kZero * scale) ++growing;
        if (l.imag() < -kZero * scale) continue;   // the other of a pair
        Mode m;
        m.eigenvalue = l;
        const Real wn = std::abs(l);
        m.natural_frequency_hz = wn / kTwoPi;
        m.damped_frequency_hz = std::abs(l.imag()) / kTwoPi;
        m.damping_ratio = wn > kZero * scale ? -l.real() / wn : 0.0;
        m.shape = L.N.cast<std::complex<Real>>() * V.col(i).head(d);
        m.largest_at = largest_entry(m.shape);
        if (m.largest_at >= 0) m.shape /= m.shape(m.largest_at);
        L.modes.push_back(std::move(m));
    }
    std::sort(L.modes.begin(), L.modes.end(), [](const Mode& a, const Mode& b) {
        return a.natural_frequency_hz < b.natural_frequency_hz;
    });

    // Undamped modes from the symmetric part of the stiffness.
    const MatX Ks = 0.5 * (L.K + L.K.transpose());
    const Eigen::GeneralizedSelfAdjointEigenSolver<MatX> ge(Ks, L.M);
    const VecX& w2 = ge.eigenvalues();
    const Real w2_scale = std::max(w2.cwiseAbs().maxCoeff(), Real(1e-300));
    int zero = 0;
    for (Index i = 0; i < w2.size(); ++i) {
        UndampedMode u;
        u.omega_squared = w2(i);
        if (std::abs(w2(i)) <= kZero * w2_scale) {
            ++zero;
            u.omega_squared = 0.0;
        }
        u.frequency_hz = u.omega_squared > 0.0 ? std::sqrt(u.omega_squared) / kTwoPi : 0.0;
        u.shape = L.N * ge.eigenvectors().col(i);
        u.largest_at = largest_entry(u.shape);
        if (u.largest_at >= 0) u.shape /= u.shape(u.largest_at);
        L.undamped.push_back(std::move(u));
    }

    if (zero > 0) {
        L.notes.push_back("MBD-K081: " + std::to_string(zero) + " of the " + std::to_string(d)
                          + " undamped modes have zero frequency: directions without stiffness "
                            "(a vehicle's position and heading on a flat road). In A they appear "
                            "as zero or real eigenvalues.");
    }
    if (growing > 0) {
        L.warnings.push_back("MBD-K082: " + std::to_string(growing)
                             + " eigenvalue(s) of A have a positive real part: motion near the "
                               "operating point grows. At an equilibrium, it is unstable.");
    }

    sim.q = q_saved;
    sim.v = v_saved;
    sim.tau = tau_saved;
    sim.refresh();
    return L;
}

std::string Linearization::summary(const Model& model) const
{
    std::ostringstream os;
    os << "linearisation: " << degrees_of_freedom() << " degrees of freedom; operating point |a| "
       << labels::number(operating_acceleration) << ", |v| " << labels::number(operating_velocity)
       << '\n';
    int k = 0;
    for (const auto& m : modes) {
        os << "mode " << ++k << ": " << labels::number(m.natural_frequency_hz) << " Hz natural, "
           << labels::number(m.damped_frequency_hz) << " Hz damped, damping ratio "
           << labels::number(m.damping_ratio);
        if (m.largest_at >= 0) os << ", largest in " << coordinate_label(model, m.largest_at);
        os << '\n';
    }
    k = 0;
    for (const auto& u : undamped) {
        os << "undamped " << ++k << ": " << labels::number(u.frequency_hz) << " Hz";
        if (u.omega_squared < 0.0) os << " (unstable, omega^2 = " << labels::number(u.omega_squared) << ")";
        if (u.largest_at >= 0) os << ", largest in " << coordinate_label(model, u.largest_at);
        os << '\n';
    }
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    for (const auto& m : notes) os << "Note: " << m << '\n';
    return os.str();
}

} // namespace mbd::kernel
