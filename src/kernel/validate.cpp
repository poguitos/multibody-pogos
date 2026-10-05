#include "mbd/kernel/validate.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/spatial/spatial.hpp"

namespace mbd::kernel {

namespace {

std::string body_label(const Model& model, int i)
{
    std::string s = "body " + std::to_string(i);
    if (i >= 0 && i < static_cast<int>(model.name.size()) && !model.name[i].empty()) {
        s += " (" + model.name[i] + ")";
    }
    return s;
}

std::string constraint_label(std::size_t k, const ConstraintModel& c)
{
    return "constraint " + std::to_string(k) + " (" + c.name() + ")";
}

std::string force_label(std::size_t k, const ForceElement& f)
{
    return "force element " + std::to_string(k) + " (" + f.name() + ")";
}

std::string number(Real x)
{
    std::ostringstream os;
    os.precision(4);
    os << x;
    return os.str();
}

bool close(const Mat6& a, const Mat6& b)
{
    return (a - b).cwiseAbs().maxCoeff() <= 1e-12 * (1.0 + b.cwiseAbs().maxCoeff());
}

bool close(const Transform3& a, const Transform3& b)
{
    return (a.p - b.p).norm() <= 1e-12 * (1.0 + b.p.norm())
        && (a.q.coeffs() - b.q.coeffs()).norm() <= 1e-12;
}

// The per-body arrays, the tree order and the joint indices. False if the
// arrays do not even have one entry per body: nothing else can be checked.
bool check_structure(const Model& model, ValidationReport& r)
{
    const std::size_t n = model.parent.size();
    if (n == 0 || model.joint.size() != n || model.X_PJ.size() != n || model.X_CJ.size() != n
        || model.inertia.size() != n || model.idx_q.size() != n || model.idx_v.size() != n
        || model.nqs.size() != n || model.nvs.size() != n || model.name.size() != n
        || model.X_JC.size() != n || model.X_CJ_motion.size() != n || model.I.size() != n) {
        r.errors.push_back("The model's per-body arrays do not all have one entry per body: "
                           "the model was changed other than through Model::add_body.");
        return false;
    }

    int nq = 0, nv = 0;
    for (int i = 1; i < model.nbodies(); ++i) {
        const std::string b = body_label(model, i);
        if (model.parent[i] < 0 || model.parent[i] >= i) {
            r.errors.push_back(b + ": its parent is body " + std::to_string(model.parent[i])
                               + ", which does not come before it. Bodies must be in topological "
                                 "order, every parent before its children.");
        }
        if (!model.joint[i]) {
            r.errors.push_back(b + ": it has no joint model.");
        } else if (model.nqs[i] != model.joint[i]->nq() || model.nvs[i] != model.joint[i]->nv()
                   || model.idx_q[i] != nq || model.idx_v[i] != nv) {
            r.errors.push_back(b + ": the coordinate and velocity indices of its joint do not "
                                   "match the joint model.");
        }
        nq += model.nqs[i];
        nv += model.nvs[i];
    }
    if (nq != model.nq || nv != model.nv) {
        r.errors.push_back("The model's totals (nq = " + std::to_string(model.nq) + ", nv = "
                           + std::to_string(model.nv) + ") do not match its joints ("
                           + std::to_string(nq) + " coordinates, " + std::to_string(nv)
                           + " velocities).");
    }
    if (!model.gravity.allFinite()) {
        r.errors.push_back("Gravity is not a finite vector.");
    }
    return true;
}

// Each body's inertia must be one a real body can have, and the quantities
// derived from it and from the joint frames when the body was added must
// still match them.
void check_bodies(const Model& model, ValidationReport& r)
{
    for (int i = 1; i < model.nbodies(); ++i) {
        const std::string b = body_label(model, i);
        const RigidBodyInertia& in = model.inertia[i];
        if (!std::isfinite(in.mass) || !in.com_B.allFinite() || !in.I_com_B.allFinite()) {
            r.errors.push_back(b + ": its mass, centre of mass or rotational inertia is not a "
                                   "finite number.");
            continue;
        }
        if (in.mass < 0.0) {
            r.errors.push_back(b + ": its mass is negative (" + number(in.mass) + " kg).");
        }

        const Mat3& J = in.I_com_B;
        const Real scale = J.cwiseAbs().maxCoeff();
        if ((J - J.transpose()).cwiseAbs().maxCoeff() > 1e-9 * scale) {
            r.errors.push_back(b + ": its rotational inertia is not symmetric.");
        } else {
            // Principal moments, ascending. A real body has none negative, and
            // the two smaller add up to at least the largest.
            const Vec3 m = Eigen::SelfAdjointEigenSolver<Mat3>(J, Eigen::EigenvaluesOnly).eigenvalues();
            const Real tol = 1e-9 * scale;
            if (m(0) < -tol) {
                r.errors.push_back(b + ": its rotational inertia has a negative principal moment ("
                                   + number(m(0)) + " kg m^2).");
            } else if (m(0) + m(1) < m(2) - tol) {
                r.errors.push_back(b + ": its principal moments of inertia (" + number(m(0)) + ", "
                                   + number(m(1)) + ", " + number(m(2))
                                   + " kg m^2) break the triangle inequality: no real body has them.");
            }
        }

        if (!close(model.I[i], spatial_inertia_matrix(in))) {
            r.errors.push_back(b + ": its inertia was changed after the body was added, and the "
                                   "algorithms still use the old one. Give the inertia to "
                                   "Model::add_body.");
        }
        if (!close(model.X_JC[i], model.X_CJ[i].inverse())
            || !close(model.X_CJ_motion[i], motion_matrix(model.X_CJ[i]))) {
            r.errors.push_back(b + ": its child-side joint frame X_CJ was changed after the body "
                                   "was added, and the algorithms still use the old one.");
        }
    }
}

// The mass matrix at q must be positive definite. When it is not, the usual
// cause is a joint that moves nothing with mass in some direction: its own
// diagonal block is then singular too.
bool check_mass_matrix(const Model& model, Data& data, const VecX& q, ValidationReport& r)
{
    if (model.nv == 0) return true;
    const MatX& M = crba(model, data, q);
    if (!M.allFinite()) {
        r.errors.push_back("The mass matrix is not finite at this configuration.");
        return false;
    }
    const Eigen::SelfAdjointEigenSolver<MatX> es(M, Eigen::EigenvaluesOnly);
    const Real largest = es.eigenvalues().cwiseAbs().maxCoeff();
    const Real threshold = 1e-12 * largest;
    if (largest > 0.0 && es.eigenvalues()(0) > threshold) return true;

    bool found = false;
    for (int i = 1; i < model.nbodies(); ++i) {
        const int nvi = model.nvs[i];
        if (nvi == 0) continue;
        const MatX block = M.block(model.idx_v[i], model.idx_v[i], nvi, nvi);
        const Eigen::SelfAdjointEigenSolver<MatX> bs(block, Eigen::EigenvaluesOnly);
        if (bs.eigenvalues()(0) <= threshold) {
            r.errors.push_back(body_label(model, i) + ": its joint can move in a direction in which "
                               "neither this body nor the bodies it carries have mass or inertia, "
                               "so the mass matrix is singular.");
            found = true;
        }
    }
    if (!found) {
        r.errors.push_back("The mass matrix is not positive definite at this configuration "
                           "(smallest eigenvalue " + number(es.eigenvalues()(0)) + ", largest "
                           + number(largest) + ").");
    }
    return false;
}

// Bodies the constraints and force elements refer to.
void check_references(const System& sys, ValidationReport& r)
{
    const Model& model = sys.model;
    const int nb = model.nbodies();
    const std::string range = "the model's bodies are 0 to " + std::to_string(nb - 1);
    for (std::size_t k = 0; k < sys.constraints.size(); ++k) {
        const auto& c = sys.constraints[k];
        if (!c) {
            r.errors.push_back("Constraint " + std::to_string(k) + " is empty (a null pointer).");
            continue;
        }
        const std::vector<int> bodies = c->bodies();
        bool in_range = true;
        for (int b : bodies) {
            if (b < 0 || b >= nb) {
                r.errors.push_back(constraint_label(k, *c) + ": it refers to body "
                                   + std::to_string(b) + ", but " + range + ".");
                in_range = false;
            }
        }
        if (in_range && bodies.size() >= 2
            && std::all_of(bodies.begin(), bodies.end(), [&](int b) { return b == bodies[0]; })) {
            r.errors.push_back(constraint_label(k, *c) + ": all its markers are on "
                               + body_label(model, bodies[0])
                               + ", so it cannot restrain anything.");
        }
    }

    for (std::size_t k = 0; k < sys.force_elements.size(); ++k) {
        const auto& f = sys.force_elements[k];
        if (!f) {
            r.errors.push_back("Force element " + std::to_string(k) + " is empty (a null pointer).");
            continue;
        }
        for (BodyIndex b : f->bodies()) {
            if (b < 0 || b >= nb) {
                r.errors.push_back(force_label(k, *f) + ": it refers to body " + std::to_string(b)
                                   + ", but " + range + ".");
            }
        }
    }
}

// A body on a free joint to the ground, with no constraint or force element
// acting on it or on anything it carries, simply falls: usually a connection
// was forgotten.
void check_floating(const System& sys, ValidationReport& r)
{
    const Model& model = sys.model;
    std::vector<char> held(static_cast<std::size_t>(model.nbodies()), 0);
    auto mark = [&](int b) {
        if (b > 0 && b < model.nbodies()) held[static_cast<std::size_t>(b)] = 1;
    };
    for (const auto& c : sys.constraints) {
        if (c) for (int b : c->bodies()) mark(b);
    }
    for (const auto& f : sys.force_elements) {
        if (f) for (BodyIndex b : f->bodies()) mark(b);
    }
    for (int i = model.nbodies() - 1; i >= 1; --i) {
        if (held[static_cast<std::size_t>(i)]) mark(model.parent[i]);
    }
    for (int i = 1; i < model.nbodies(); ++i) {
        if (model.parent[i] == 0 && model.joint[i] && std::string(model.joint[i]->name()) == "free"
            && !held[static_cast<std::size_t>(i)]) {
            r.notes.push_back(body_label(model, i) + " floats freely: it is on a free joint to the "
                              "ground, and no constraint or force element acts on it or on the "
                              "bodies it carries.");
        }
    }
}

// Residuals and rank of the constraints at (q, t).
void check_constraints(const System& sys, Data& data, const VecX& q, Real t, ValidationReport& r)
{
    const Model& model = sys.model;
    ConstraintSolver solver(model, sys.constraints);
    r.constraint_equations = solver.size();
    if (solver.size() == 0) {
        r.independent_equations = 0;
        return;
    }

    const VecX zero_v = VecX::Zero(model.nv);
    solver.evaluate(data, q, zero_v, t);
    Real worst = 0.0;
    std::size_t worst_k = 0;
    Index row = 0;
    for (std::size_t k = 0; k < sys.constraints.size(); ++k) {
        const Index m = sys.constraints[k]->size();
        if (m == 0) continue;
        const Real residual = solver.phi().segment(row, m).cwiseAbs().maxCoeff();
        if (residual > worst) {
            worst = residual;
            worst_k = k;
        }
        row += m;
    }
    if (worst > 1e-6) {
        r.warnings.push_back("The constraints are not satisfied at this configuration: the largest "
                             "residual, " + number(worst) + ", is in "
                             + constraint_label(worst_k, *sys.constraints[worst_k])
                             + ". Simulator::initialize moves the bodies onto the constraints.");
    }

    // The rank the solver itself will see: that of J M^-1 J^T, with the same
    // tolerance.
    solver.forward_dynamics(data, q, zero_v, zero_v, t);
    r.independent_equations = solver.info().rank;
    r.redundant_equations = solver.info().equations - solver.info().rank;
    if (r.redundant_equations > 0) {
        r.notes.push_back(std::to_string(r.redundant_equations) + " of the "
                          + std::to_string(r.constraint_equations)
                          + " constraint equations are redundant at this configuration (a loop "
                            "closed twice, or a planar loop closed in three dimensions). The "
                            "solver drops them and gives them no force.");
    }
}

// The checks in order: the numerical ones only once the structure is sound.
// q == nullptr means the neutral configuration.
ValidationReport run_checks(const System& sys, const VecX* q_given, Real t)
{
    const Model& model = sys.model;
    ValidationReport r;
    r.bodies = std::max(model.nbodies() - 1, 0);
    r.coordinates = model.nq;
    r.velocities = model.nv;
    r.degrees_of_freedom = model.nv;

    if (!check_structure(model, r)) return r;
    check_bodies(model, r);
    check_references(sys, r);
    check_floating(sys, r);
    if (!r.errors.empty()) return r;

    const VecX q = q_given ? *q_given : model.neutral_configuration();
    if (q.size() != model.nq || !q.allFinite()) {
        r.errors.push_back("The configuration has " + std::to_string(q.size())
                           + " entries, or entries that are not finite; the model has nq = "
                           + std::to_string(model.nq) + ".");
        return r;
    }

    Data data(model);
    if (check_mass_matrix(model, data, q, r)) {
        check_constraints(sys, data, q, t, r);
    }
    r.degrees_of_freedom = r.velocities - r.independent_equations;
    return r;
}

} // namespace

std::string ValidationReport::summary() const
{
    std::ostringstream os;
    os << "bodies " << bodies << ", coordinates " << coordinates << ", velocities " << velocities
       << "; constraint equations " << constraint_equations << " (" << independent_equations
       << " independent, " << redundant_equations << " redundant); degrees of freedom "
       << degrees_of_freedom << '\n';
    for (const auto& m : errors) os << "Error: " << m << '\n';
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    for (const auto& m : notes) os << "Note: " << m << '\n';
    return os.str();
}

ValidationReport validate(const System& sys)
{
    return run_checks(sys, nullptr, 0.0);
}

ValidationReport validate(const System& sys, const VecX& q, Real t)
{
    return run_checks(sys, &q, t);
}

} // namespace mbd::kernel
