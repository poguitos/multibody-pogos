#include "mbd/kernel/describe.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/validate.hpp"

#include "labels.hpp"

namespace mbd::kernel {

namespace {

using detail::body_label;

std::string vec3(const Vec3& x)
{
    std::ostringstream os;
    os.precision(6);
    os << '(' << x.x() << ", " << x.y() << ", " << x.z() << ')';
    return os.str();
}

/// Each body once, in the order first met (a constraint set lists its
/// parts' bodies one after another).
std::string bodies_of(const Model& model, const std::vector<int>& bodies)
{
    std::string s;
    std::vector<int> seen;
    for (int b : bodies) {
        if (std::find(seen.begin(), seen.end(), b) != seen.end()) continue;
        seen.push_back(b);
        s += (s.empty() ? "" : ", ") + (b == 0 ? std::string("body 0 (ground)") : body_label(model, b));
    }
    return s;
}

/// "0" or "2-4": a range of indices of the given length.
std::string range(int first, int count)
{
    if (count == 0) return "none";
    if (count == 1) return std::to_string(first);
    return std::to_string(first) + "-" + std::to_string(first + count - 1);
}

} // namespace

namespace {

std::string describe_with(const System& sys, const ValidationReport& v)
{
    const Model& model = sys.model;
    std::ostringstream os;
    os.precision(6);
    os << "System: " << model.nbodies() - 1 << " bodies, " << model.nq << " coordinates, " << model.nv
       << " velocities; " << sys.constraints.size() << " constraints (" << v.constraint_equations
       << " equations, rank " << v.independent_equations << "); " << sys.force_elements.size()
       << " force elements, " << sys.joint_forces.size() << " joint forces; degrees of freedom "
       << v.degrees_of_freedom << ".\n"
       << "validate(): " << v.errors.size() << " errors, " << v.warnings.size() << " warnings, "
       << v.notes.size() << " notes.\n"
       << "Gravity " << vec3(model.gravity) << " m/s^2.\n"
       << "Bodies:\n";
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto b = static_cast<std::size_t>(i);
        const RigidBodyInertia& I = model.inertia[b];
        const int parent = model.parent[b];
        os << "  " << body_label(model, i) << ": " << model.joint[b]->name() << " joint to "
           << (parent == 0 ? std::string("body 0 (ground)") : body_label(model, parent)) << "; coordinates "
           << range(model.idx_q[b], model.nqs[b]) << ", velocities " << range(model.idx_v[b], model.nvs[b])
           << "; mass " << I.mass << " kg, centre of mass " << vec3(I.com_B) << " in the body frame\n";
    }
    os << "Constraints:" << (sys.constraints.empty() ? " none" : "") << '\n';
    for (std::size_t k = 0; k < sys.constraints.size(); ++k) {
        const auto& c = sys.constraints[k];
        os << "  " << k << " " << c->name() << ": " << c->size() << " equations, on "
           << bodies_of(model, c->bodies()) << '\n';
    }
    os << "Force elements:" << (sys.force_elements.empty() ? " none" : "") << '\n';
    for (std::size_t k = 0; k < sys.force_elements.size(); ++k) {
        const auto& f = sys.force_elements[k];
        os << "  " << k << " " << f->name() << ": on " << bodies_of(model, f->bodies()) << '\n';
    }
    os << "Joint forces:" << (sys.joint_forces.empty() ? " none" : "") << '\n';
    for (std::size_t k = 0; k < sys.joint_forces.size(); ++k) {
        const auto& f = sys.joint_forces[k];
        os << "  " << k << " " << f->name() << ": on the joint of " << bodies_of(model, f->bodies()) << '\n';
    }
    return os.str();
}

} // namespace

std::string describe(const System& sys)
{
    return describe_with(sys, validate(sys));
}

std::string describe(const System& sys, const VecX& q, Real t)
{
    return describe_with(sys, validate(sys, q, t));
}

std::string dump_state(const Simulator& sim)
{
    const Model& model = sim.system.model;
    std::ostringstream os;
    os.precision(10);
    os << "State at t = " << sim.time << " s\n  q =";
    for (Index k = 0; k < sim.q.size(); ++k) os << ' ' << sim.q(k);
    os << "\n  v =";
    for (Index k = 0; k < sim.v.size(); ++k) os << ' ' << sim.v(k);
    os << "\nBodies, in world axes (position, orientation as w x y z, velocity, angular velocity):\n";
    for (int i = 1; i < model.nbodies(); ++i) {
        const RigidBodyState& s = sim.states()[static_cast<std::size_t>(i)];
        os << "  " << body_label(model, i) << ": " << vec3(s.p_WB) << ", (" << s.q_WB.w() << ", " << s.q_WB.x()
           << ", " << s.q_WB.y() << ", " << s.q_WB.z() << "), " << vec3(s.v_WB) << ", " << vec3(s.w_WB) << '\n';
    }
    if (!sim.system.constraints.empty()) {
        ConstraintSolver solver(model, sim.system.constraints);
        Data data(model);
        solver.evaluate(data, sim.q, sim.v, sim.time);
        os << "Constraints: |phi| = " << solver.phi().norm()
           << ", |J v - nu| = " << (solver.J() * sim.v - solver.nu()).norm() << '\n';
    }
    return os.str();
}

} // namespace mbd::kernel
