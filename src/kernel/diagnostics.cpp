#include "mbd/kernel/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/validate.hpp"

#include "labels.hpp"

namespace mbd::kernel {

namespace {

std::string vec3(const Vec3& x)
{
    std::ostringstream os;
    os.precision(6);
    os << '(' << x.x() << ", " << x.y() << ", " << x.z() << ')';
    return os.str();
}

// Each body once, in the order first met (a closure lists its bodies once
// per primitive).
std::string bodies_of(const Model& model, const std::vector<int>& bodies)
{
    std::string s;
    std::vector<int> seen;
    for (int b : bodies) {
        if (std::find(seen.begin(), seen.end(), b) != seen.end()) continue;
        s += (seen.empty() ? "" : ", ") + labels::body(model, b);
        seen.push_back(b);
    }
    return s;
}

std::string range(int first, int n)
{
    if (n == 0) return "none";
    if (n == 1) return std::to_string(first);
    return std::to_string(first) + "-" + std::to_string(first + n - 1);
}

// The trace's columns, in the order written.
const char* const kColumns[] = {"time", "dt", "drift", "projection_iterations", "position_residual",
                                "velocity_residual", "projection_converged", "equations", "rank",
                                "kinetic", "potential", "power", "work", "balance", "max_speed",
                                "events"};
constexpr std::size_t kColumnCount = sizeof(kColumns) / sizeof(kColumns[0]);

} // namespace

namespace {

std::string describe_with(const System& sys, const ValidationReport& r, const char* where)
{
    const Model& model = sys.model;
    std::ostringstream os;
    os << "model: " << model.nbodies() - 1 << " bodies besides the ground, " << model.nq
       << " coordinates, " << model.nv << " velocities; gravity " << vec3(model.gravity) << '\n';
    os << "bodies:\n";
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto k = static_cast<std::size_t>(i);
        const auto& joint = model.joint[k];
        const RigidBodyInertia& I = model.inertia[k];
        os << "  " << labels::body(model, i) << ": " << (joint ? joint->name() : "no") << " joint on "
           << labels::body(model, model.parent[k]) << "; q " << range(model.idx_q[k], model.nqs[k])
           << ", v " << range(model.idx_v[k], model.nvs[k]) << "; mass " << labels::number(I.mass)
           << " kg, centre of mass " << vec3(I.com_B) << '\n';
    }
    os << "constraints: " << sys.constraints.size() << '\n';
    for (std::size_t k = 0; k < sys.constraints.size(); ++k) {
        const auto& c = sys.constraints[k];
        os << "  " << labels::constraint(k, *c) << ", " << c->size() << " equation(s), on "
           << bodies_of(model, c->bodies()) << '\n';
    }
    os << "force elements: " << sys.force_elements.size() << '\n';
    for (std::size_t k = 0; k < sys.force_elements.size(); ++k) {
        const auto& f = sys.force_elements[k];
        std::vector<int> b;
        for (BodyIndex x : f->bodies()) b.push_back(static_cast<int>(x));
        os << "  force element " << k << " (" << f->name() << "), on " << bodies_of(model, b) << '\n';
    }
    os << "joint forces: " << sys.joint_forces.size() << '\n';
    for (std::size_t k = 0; k < sys.joint_forces.size(); ++k) {
        const auto& f = sys.joint_forces[k];
        os << "  joint force " << k << " (" << f->name() << "), on " << bodies_of(model, f->bodies()) << '\n';
    }
    os << "degrees of freedom: " << r.velocities << " velocities less " << r.independent_equations
       << " independent constraint equations = " << r.degrees_of_freedom << " (" << where
       << "); validate(): " << r.errors.size() << " error(s), " << r.warnings.size() << " warning(s), "
       << r.notes.size() << " note(s)\n";
    return os.str();
}

} // namespace

std::string describe(const System& sys)
{
    return describe_with(sys, validate(sys), "at the neutral configuration");
}

std::string describe(const System& sys, const VecX& q, Real t)
{
    return describe_with(sys, validate(sys, q, t), "at the configuration given");
}

std::string dump_state(const Simulator& sim)
{
    const Model& model = sim.system.model;
    std::ostringstream os;
    os.precision(10);
    os << "time " << sim.time << " s\n";
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto k = static_cast<std::size_t>(i);
        const RigidBodyState& s = sim.states()[k];
        os << labels::body(model, i) << ": q (";
        for (int j = 0; j < model.nqs[k]; ++j) os << (j ? ", " : "") << sim.q(model.idx_q[k] + j);
        os << "), v (";
        for (int j = 0; j < model.nvs[k]; ++j) os << (j ? ", " : "") << sim.v(model.idx_v[k] + j);
        os << "); at " << vec3(s.p_WB) << ", quaternion (" << s.q_WB.x() << ", " << s.q_WB.y() << ", "
           << s.q_WB.z() << ", " << s.q_WB.w() << "), velocity " << vec3(s.v_WB) << ", angular "
           << vec3(s.w_WB) << '\n';
    }
    if (!sim.system.constraints.empty()) {
        Data data(model);
        ConstraintSolver solver(model, sim.system.constraints);
        solver.evaluate(data, sim.q, sim.v, sim.time);
        Index row = 0;
        for (std::size_t k = 0; k < sim.system.constraints.size(); ++k) {
            const auto& c = sim.system.constraints[k];
            const Index m = c->size();
            const Real worst = m > 0 ? solver.phi().segment(row, m).cwiseAbs().maxCoeff() : 0.0;
            os << labels::constraint(k, *c) << ": largest |phi| " << labels::number(worst) << '\n';
            row += m;
        }
        const ProjectionInfo& p = sim.last_projection();
        os << "last projection: " << p.iterations << " iterations, |phi| "
           << labels::number(p.position_residual) << ", |J v - nu| " << labels::number(p.velocity_residual)
           << (p.converged ? ", converged" : ", NOT converged") << "; failures so far "
           << sim.projection_failures() << '\n';
    }
    return os.str();
}

void write_trace_csv(const std::vector<TraceRow>& trace, const std::string& path)
{
    std::ofstream out(path);
    MBD_THROW_IF(!out, "MBD-A030: write_trace_csv: cannot open " + path + " for writing");
    for (std::size_t k = 0; k < kColumnCount; ++k) out << (k ? "," : "") << kColumns[k];
    out << '\n' << std::setprecision(17);
    for (const TraceRow& r : trace) {
        out << r.time << ',' << r.dt << ',' << r.drift << ',' << r.projection_iterations << ','
            << r.position_residual << ',' << r.velocity_residual << ',' << (r.projection_converged ? 1 : 0)
            << ',' << r.equations << ',' << r.rank << ',' << r.kinetic << ',' << r.potential << ','
            << r.power << ',' << r.work << ',' << r.balance << ',' << r.max_speed << ',' << r.events
            << '\n';
    }
}

std::vector<TraceRow> read_trace_csv(const std::string& path)
{
    std::ifstream in(path);
    MBD_THROW_IF(!in, "MBD-A030: read_trace_csv: cannot open " + path);
    std::string line;
    std::string header;
    for (std::size_t k = 0; k < kColumnCount; ++k) header += std::string(k ? "," : "") + kColumns[k];
    MBD_THROW_IF(!std::getline(in, line) || line != header,
                 "MBD-A030: read_trace_csv: " + path + " does not start with the trace's header");
    std::vector<TraceRow> trace;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<Real> x;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) x.push_back(std::stod(field));
        MBD_THROW_IF(x.size() != kColumnCount,
                     "MBD-A030: read_trace_csv: a row of " + path + " has " + std::to_string(x.size())
                         + " fields, not " + std::to_string(kColumnCount));
        TraceRow r;
        r.time = x[0];
        r.dt = x[1];
        r.drift = x[2];
        r.projection_iterations = static_cast<int>(x[3]);
        r.position_residual = x[4];
        r.velocity_residual = x[5];
        r.projection_converged = x[6] != 0.0;
        r.equations = static_cast<int>(x[7]);
        r.rank = static_cast<int>(x[8]);
        r.kinetic = x[9];
        r.potential = x[10];
        r.power = x[11];
        r.work = x[12];
        r.balance = x[13];
        r.max_speed = x[14];
        r.events = static_cast<int>(x[15]);
        trace.push_back(r);
    }
    return trace;
}

std::string TraceDiagnosis::summary() const
{
    std::ostringstream os;
    if (clean()) os << "trace: nothing to report\n";
    for (const auto& m : warnings) os << "Warning: " << m << '\n';
    for (const auto& m : notes) os << "Note: " << m << '\n';
    return os.str();
}

TraceDiagnosis diagnose(const std::vector<TraceRow>& trace)
{
    TraceDiagnosis d;
    if (trace.size() < 2) return d;

    // Projections that failed.
    int failed = 0;
    Real first_failure = 0.0, worst_residual = 0.0;
    for (const TraceRow& r : trace) {
        if (r.projection_converged) continue;
        if (failed++ == 0) first_failure = r.time;
        worst_residual = std::max(worst_residual, r.position_residual);
    }
    if (failed > 0) {
        d.warnings.push_back("MBD-K100: The projection onto the constraints failed in " + std::to_string(failed)
                             + " of " + std::to_string(trace.size() - 1) + " steps, first at t = "
                             + labels::number(first_failure) + " s, leaving |phi| up to "
                             + labels::number(worst_residual)
                             + ": the constraints cannot be met there (a loop that cannot close, "
                               "contradictory drivers, a singular position).");
    }

    // The scale of the energy that changes hands during the run.
    const TraceRow& first = trace.front();
    Real scale = 0.0;
    for (const TraceRow& r : trace) {
        scale = std::max({scale, r.kinetic, std::abs(r.potential - first.potential), std::abs(r.work)});
    }
    if (scale > 0.0) {
        // The energy balance, without the jumps of event actions (an impact
        // that loses energy by design).
        Real drift = 0.0, worst = 0.0, worst_time = 0.0;
        for (std::size_t k = 1; k < trace.size(); ++k) {
            if (trace[k].events == 0) drift += trace[k].balance - trace[k - 1].balance;
            if (std::abs(drift) > std::abs(worst)) {
                worst = drift;
                worst_time = trace[k].time;
            }
        }
        if (std::abs(worst) > 1e-4 * scale) {
            d.warnings.push_back("MBD-K101: The energy balance drifts by " + labels::number(worst) + " J ("
                                 + labels::number(std::abs(worst) / scale)
                                 + " of the energy exchanged), largest at t = " + labels::number(worst_time)
                                 + " s: the integration or the projection makes or loses energy. "
                                   "Usually the step is too long for the fastest motion.");
        }
        // Energy put in by the applied forces. The energy here is kinetic and
        // gravity's: a spring's store is in the work, so a spring that
        // releases energy makes it rise too, and back. What it cannot do is
        // raise the peaks: compare the largest energy of the run's last
        // quarter with that of its first.
        const std::size_t quarter = std::max<std::size_t>(trace.size() / 4, 1);
        Real early = -1e300, late = -1e300;
        for (std::size_t k = 0; k < quarter; ++k) {
            early = std::max(early, trace[k].kinetic + trace[k].potential);
            const TraceRow& r = trace[trace.size() - 1 - k];
            late = std::max(late, r.kinetic + r.potential);
        }
        const TraceRow& last = trace.back();
        if (last.work > 1e-3 * scale && late - early > 1e-3 * scale) {
            d.notes.push_back("MBD-K102: The energy's peaks grew by " + labels::number(late - early)
                              + " J from the first quarter of the run to the last, and the applied "
                                "forces did " + labels::number(last.work)
                              + " J of work: a driver or a motor does that, and so does a force "
                                "element with the wrong sign (a damper that pushes). Over less "
                                "than a period it can also be a spring letting go of its energy.");
        }
    }

    // Redundancy and drift within a step.
    int redundant_from = -1;
    Real worst_drift = 0.0, drift_time = 0.0;
    for (std::size_t k = 0; k < trace.size(); ++k) {
        if (redundant_from < 0 && trace[k].rank < trace[k].equations) redundant_from = static_cast<int>(k);
        if (trace[k].drift > worst_drift) {
            worst_drift = trace[k].drift;
            drift_time = trace[k].time;
        }
    }
    if (redundant_from >= 0) {
        const TraceRow& r = trace[static_cast<std::size_t>(redundant_from)];
        d.notes.push_back("MBD-K103: " + std::to_string(r.equations - r.rank) + " of the " + std::to_string(r.equations)
                          + " constraint equations are redundant, from t = " + labels::number(r.time)
                          + " s (see MBD-K050).");
    }
    if (worst_drift > 1e-6) {
        d.warnings.push_back("MBD-K104: Within a step the constraints drift by up to " + labels::number(worst_drift)
                             + " before the projection (at t = " + labels::number(drift_time)
                             + " s): the step is long for the motion, and the dynamics within it "
                               "were computed off the constraints.");
    }
    return d;
}

} // namespace mbd::kernel
