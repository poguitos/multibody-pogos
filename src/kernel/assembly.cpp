#include "mbd/kernel/assembly.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"

#include "checks.hpp"
#include "labels.hpp"

namespace mbd::kernel {

namespace {

void add_joint_coordinate(const Model& model, int body, int k, AssemblySpec& spec)
{
    const int index = model.idx_v[static_cast<std::size_t>(body)] + k;
    spec.hold_positions.push_back(index);
    spec.hold_velocities.push_back(index);
    spec.hold_accelerations.push_back(index);
}

void check_jointed_body(const Model& model, int body)
{
    MBD_THROW_IF(body < 1 || body >= model.nbodies(),
                 "MBD-K060: kernel::AssemblySpec::hold: there is no jointed body " + std::to_string(body)
                     + "; the bodies with joints are 1 to " + std::to_string(model.nbodies() - 1) + ".");
}

// The held list of one level, sorted and without repeats. Throws if an index
// is out of range or holds part of a rotation (see assembly.hpp).
std::vector<int> checked_held(const Model& model, const std::vector<int>& given, const char* level)
{
    std::vector<int> held = given;
    std::sort(held.begin(), held.end());
    held.erase(std::unique(held.begin(), held.end()), held.end());
    for (int k : held) {
        MBD_THROW_IF(k < 0 || k >= model.nv,
                     std::string("MBD-K060: kernel::assemble: the held ") + level + " include "
                         + std::to_string(k) + ", but the velocity coordinates are 0 to "
                         + std::to_string(model.nv - 1) + ".");
    }
    auto count = [&](int first, int n) {
        return static_cast<int>(std::count_if(held.begin(), held.end(),
                                              [&](int k) { return k >= first && k < first + n; }));
    };
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto& joint = model.joint[static_cast<std::size_t>(i)];
        if (!joint) continue;
        const char* name = joint->name();
        const bool is_spherical = std::strcmp(name, "spherical") == 0;
        const bool is_free = std::strcmp(name, "free") == 0;
        if (!is_spherical && !is_free) continue;
        const int first = model.idx_v[static_cast<std::size_t>(i)];
        const int angular = count(first, 3);
        const int linear = is_free ? count(first + 3, 3) : 0;
        MBD_THROW_IF((angular != 0 && angular != 3) || (angular == 0 && linear != 0 && linear != 3),
                     std::string("MBD-K061: kernel::assemble: the held ") + level + " hold part of the "
                         + name + " joint of " + labels::body(model, i) + " ("
                         + std::to_string(angular) + " of its 3 angular coordinates, "
                         + std::to_string(linear)
                         + " of its linear ones). Hold its rotation whole, or not at all; its "
                           "translation may be held in part only when the rotation is held.");
    }
    return held;
}

// Largest entry of |x| and its index; -1 if x is zero.
void largest(const VecX& x, AssemblyLevel& level)
{
    level.largest_change = 0.0;
    level.largest_change_at = -1;
    for (Index k = 0; k < x.size(); ++k) {
        if (std::abs(x(k)) > level.largest_change) {
            level.largest_change = std::abs(x(k));
            level.largest_change_at = static_cast<int>(k);
        }
    }
}

// The constraint with the largest |phi| in the solver's last evaluation.
std::string worst_constraint(const System& sys, const ConstraintSolver& solver)
{
    Real worst = -1.0;
    std::size_t worst_k = 0;
    Index row = 0;
    for (std::size_t k = 0; k < sys.constraints.size(); ++k) {
        const Index m = sys.constraints[k]->size();
        if (m > 0) {
            const Real r = solver.phi().segment(row, m).cwiseAbs().maxCoeff();
            if (r > worst) {
                worst = r;
                worst_k = k;
            }
        }
        row += m;
    }
    return labels::constraint(worst_k, *sys.constraints[worst_k]) + ", " + labels::number(worst);
}

// The warning and the note that follow from a level's ranks. The note is
// only worth giving when the level changed something.
void judge_level(const char* level, int rank_all, AssemblyLevel& l, AssemblyReport& r)
{
    const int lost = rank_all - l.independent_equations;
    if (lost > 0) {
        r.warnings.push_back("MBD-K065: The " + std::to_string(l.held) + " held " + level + " take "
                             + std::to_string(lost) + " of the " + std::to_string(rank_all)
                             + " independent constraint directions away from the coordinates left "
                               "free: the held values must satisfy "
                             + std::to_string(lost) + " constraint equation(s) by themselves.");
    }
    if (l.freedom_left > 0 && l.largest_change_at >= 0) {
        r.notes.push_back("MBD-K066: The held " + std::string(level) + " leave "
                          + std::to_string(l.freedom_left)
                          + " degree(s) of freedom; along them the assembly changed the given "
                          + level + " as little as possible, in the kinetic-energy metric.");
    }
}

AssemblyReport run_assembly(const System& sys, VecX& q, VecX& v, VecX* a, Real t,
                            const AssemblySpec& spec)
{
    const Model& model = sys.model;
    checks::q("kernel::assemble", model, q);
    checks::v("kernel::assemble", model, v);
    if (a) checks::v("kernel::assemble", model, *a, "a");
    const std::vector<int> held_q = checked_held(model, spec.hold_positions, "positions");
    const std::vector<int> held_v = checked_held(model, spec.hold_velocities, "velocities");
    const std::vector<int> held_a =
        a ? checked_held(model, spec.hold_accelerations, "accelerations") : std::vector<int>{};

    Data data(model);
    ConstraintSolver solver(model, sys.constraints);
    const VecX zero_v = VecX::Zero(model.nv);
    VecX change(model.nv);

    AssemblyReport r;
    r.velocities = model.nv;
    r.constraint_equations = solver.size();

    // Positions.
    AssemblyLevel& p = r.position;
    p.held = static_cast<int>(held_q.size());
    solver.evaluate(data, q, zero_v, t);
    p.residual_before = solver.phi().norm();
    const VecX q_given = q;
    const AssemblyStepInfo pi =
        solver.assemble_positions(data, q, t, held_q, spec.tolerance, spec.max_iterations);
    p.done = true;
    p.residual = pi.residual;
    p.iterations = pi.iterations;
    p.refinements = pi.refinements;
    p.independent_equations = pi.rank;
    p.freedom_left = model.nv - p.held - pi.rank;
    p.converged = pi.converged;
    difference(model, q_given, q, change);
    largest(change, p);

    // The rank of all the constraints, as the solver will see it in a
    // simulation (validate() counts it the same way).
    if (solver.size() > 0) {
        solver.forward_dynamics(data, q, zero_v, zero_v, t);
        r.independent_equations = solver.info().rank;
    }
    r.degrees_of_freedom = model.nv - r.independent_equations;
    judge_level("positions", r.independent_equations, p, r);
    if (!p.converged) {
        solver.evaluate(data, q, zero_v, t);
        r.errors.push_back("MBD-K062: The positions could not be assembled: |phi| is "
                           + labels::number(p.residual) + " after " + std::to_string(p.iterations)
                           + " Gauss-Newton steps (tolerance " + labels::number(spec.tolerance)
                           + "); the largest residual is in " + worst_constraint(sys, solver)
                           + ". The velocities were left as given.");
        return r;
    }

    // Velocities, at the assembled positions.
    AssemblyLevel& vl = r.velocity;
    vl.held = static_cast<int>(held_v.size());
    if (solver.size() > 0) {
        solver.evaluate(data, q, zero_v, t);
        VecX rhs = solver.nu();
        rhs.noalias() -= solver.J() * v;
        vl.residual_before = rhs.norm();
    }
    const VecX v_given = v;
    const AssemblyStepInfo vi = solver.assemble_velocities(data, q, v, t, held_v, spec.tolerance);
    vl.done = true;
    vl.residual = vi.residual;
    vl.independent_equations = vi.rank;
    vl.freedom_left = model.nv - vl.held - vi.rank;
    vl.converged = vi.converged;
    change = v - v_given;
    largest(change, vl);
    judge_level("velocities", r.independent_equations, vl, r);
    if (!vl.converged) {
        r.errors.push_back("MBD-K063: The velocities could not be assembled: |J v - nu| is "
                           + labels::number(vl.residual) + " with the velocities held (tolerance "
                           + labels::number(spec.tolerance) + ").");
        return r;
    }
    if (!a) return r;

    // Accelerations, at the assembled positions and velocities.
    AssemblyLevel& al = r.acceleration;
    al.held = static_cast<int>(held_a.size());
    if (solver.size() > 0) {
        solver.evaluate(data, q, v, t);
        VecX rhs = solver.gamma();
        rhs.noalias() -= solver.J() * (*a);
        al.residual_before = rhs.norm();
    }
    const VecX a_given = *a;
    const AssemblyStepInfo ai =
        solver.assemble_accelerations(data, q, v, *a, t, held_a, spec.tolerance);
    al.done = true;
    al.residual = ai.residual;
    al.independent_equations = ai.rank;
    al.freedom_left = model.nv - al.held - ai.rank;
    al.converged = ai.converged;
    change = *a - a_given;
    largest(change, al);
    judge_level("accelerations", r.independent_equations, al, r);
    if (!al.converged) {
        r.errors.push_back("MBD-K064: The accelerations could not be assembled: |J a - gamma| is "
                           + labels::number(al.residual) + " with the accelerations held (tolerance "
                           + labels::number(spec.tolerance) + ").");
    }
    return r;
}

void level_line(std::ostringstream& os, const Model& model, const char* name,
                const char* residual, const AssemblyLevel& l)
{
    os << name << ": ";
    if (!l.done) {
        os << "not assembled\n";
        return;
    }
    os << l.held << " held, " << l.freedom_left << " degree(s) of freedom left; " << residual << ' '
       << labels::number(l.residual_before) << " -> " << labels::number(l.residual);
    if (l.iterations > 0) {
        os << " in " << l.iterations << " Gauss-Newton steps (" << l.refinements
           << " towards the least correction)";
    }
    if (l.largest_change_at >= 0) {
        os << "; largest change " << labels::number(l.largest_change) << " in "
           << coordinate_label(model, l.largest_change_at);
    } else {
        os << "; nothing changed";
    }
    os << (l.converged ? "" : "; NOT CONVERGED") << '\n';
}

} // namespace

AssemblySpec& AssemblySpec::hold(const Model& model, int body)
{
    check_jointed_body(model, body);
    for (int k = 0; k < model.nvs[static_cast<std::size_t>(body)]; ++k) {
        add_joint_coordinate(model, body, k, *this);
    }
    return *this;
}

AssemblySpec& AssemblySpec::hold(const Model& model, int body, int k)
{
    check_jointed_body(model, body);
    const int n = model.nvs[static_cast<std::size_t>(body)];
    MBD_THROW_IF(k < 0 || k >= n,
                 "MBD-K060: kernel::AssemblySpec::hold: the joint of " + labels::body(model, body)
                     + " has coordinates 0 to " + std::to_string(n - 1) + ", not "
                     + std::to_string(k) + ".");
    add_joint_coordinate(model, body, k, *this);
    return *this;
}

std::string coordinate_label(const Model& model, int k)
{
    for (int i = 1; i < model.nbodies(); ++i) {
        const int first = model.idx_v[static_cast<std::size_t>(i)];
        const int n = model.nvs[static_cast<std::size_t>(i)];
        if (k >= first && k < first + n) {
            const auto& joint = model.joint[static_cast<std::size_t>(i)];
            const std::string joint_name = joint ? joint->name() : "unknown";
            std::string s = "v[" + std::to_string(k) + "], ";
            if (n > 1) s += "coordinate " + std::to_string(k - first) + " of ";
            return s + "the " + joint_name + " joint of " + labels::body(model, i);
        }
    }
    return "v[" + std::to_string(k) + "]";
}

std::string AssemblyReport::summary(const Model& model) const
{
    std::ostringstream os;
    os << "assembly: velocities " << velocities << "; constraint equations " << constraint_equations
       << " (" << independent_equations << " independent); degrees of freedom "
       << degrees_of_freedom << '\n';
    level_line(os, model, "positions", "|phi|", position);
    level_line(os, model, "velocities", "|J v - nu|", velocity);
    level_line(os, model, "accelerations", "|J a - gamma|", acceleration);
    for (const auto& m : errors) os << "Error: " << m << '\n';
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    for (const auto& m : notes) os << "Note: " << m << '\n';
    return os.str();
}

AssemblyReport assemble(const System& sys, VecX& q, VecX& v, Real t, const AssemblySpec& spec)
{
    return run_assembly(sys, q, v, nullptr, t, spec);
}

AssemblyReport assemble(const System& sys, VecX& q, VecX& v, VecX& a, Real t,
                        const AssemblySpec& spec)
{
    return run_assembly(sys, q, v, &a, t, spec);
}

AssemblyReport assemble(Simulator& sim, const AssemblySpec& spec)
{
    AssemblyReport r = run_assembly(sim.system, sim.q, sim.v, nullptr, sim.time, spec);
    sim.refresh();
    if (!r.ok()) {
        std::string all;
        for (const auto& e : r.errors) all += (all.empty() ? "" : " ") + e;
        report_warning("MBD-K067: Simulator assembly at t = " + labels::number(sim.time)
                       + " s did not succeed; the state is the closest it reached. " + all);
    }
    return r;
}

} // namespace mbd::kernel
