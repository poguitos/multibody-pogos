// Timing of the dynamics algorithms (plan task 2.10), of the integrators on
// the sedan (task 4.5), and of the mass matrix factorized densely and along
// the tree (task 4.8).
//
// Built on request only:  scripts\build.ps1 -Target mbd_bench
// Run:                    build\bench\mbd_bench.exe
//
// Prints a Markdown table of the time per call, the median of several
// rounds. Numbers are machine-dependent: compare runs on the same machine.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/tree_cholesky.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

using namespace mbd;

namespace
{
    // Median time per call in microseconds. Each round runs `f` enough times
    // to last about 20 ms.
    double time_per_call_us(const std::function<void()>& f)
    {
        using clock = std::chrono::steady_clock;
        f();  // warm-up

        int reps = 1;
        for (;;) {
            const auto t0 = clock::now();
            for (int i = 0; i < reps; ++i) f();
            const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
            if (ms > 20.0 || reps > (1 << 22)) break;
            reps *= 2;
        }

        std::vector<double> rounds;
        for (int r = 0; r < 7; ++r) {
            const auto t0 = clock::now();
            for (int i = 0; i < reps; ++i) f();
            const double us = std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            rounds.push_back(us / reps);
        }
        std::sort(rounds.begin(), rounds.end());
        return rounds[rounds.size() / 2];
    }

    // Deterministic pseudo-random numbers in [lo, hi).
    struct Lcg {
        std::uint64_t s{12345};
        double range(double lo, double hi)
        {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            return lo + (hi - lo) * double(s >> 11) / 9007199254740992.0;
        }
    };

    Transform3 generic_frame(Lcg& rng, double offset)
    {
        Vec3 axis(rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1));
        if (axis.norm() < 1e-3) axis = Vec3::UnitX();
        axis.normalize();
        return Transform3(Quat(Eigen::AngleAxisd(rng.range(-1.0, 1.0), axis)),
                          Vec3(rng.range(-offset, offset), rng.range(-offset, offset),
                               rng.range(-offset, offset)));
    }

    // n bodies on revolute joints with generic axes, and a random state;
    // body i is a child of body parent_of(i).
    void build_revolute_system(kernel::Model& model, VecX& q, VecX& v, int n,
                               const std::function<int(int)>& parent_of)
    {
        Lcg rng;
        const auto revolute = std::make_shared<kernel::RevoluteJointModel>();
        for (int i = 1; i <= n; ++i) {
            const BodyIndex parent = BodyIndex(parent_of(i));
            auto inertia = RigidBodyInertia::from_solid_box(
                rng.range(0.5, 3.0), Vec3(rng.range(0.05, 0.3), rng.range(0.05, 0.3), rng.range(0.05, 0.3)));
            inertia.com_B = Vec3(rng.range(-0.1, 0.1), rng.range(-0.1, 0.1), rng.range(-0.1, 0.1));
            const Transform3 X_PJ = generic_frame(rng, 0.4);
            const Transform3 X_CJ = generic_frame(rng, 0.3);
            model.add_body(parent, revolute, X_PJ, X_CJ, inertia);
        }
        q.resize(model.nq);
        v.resize(model.nv);
        for (int i = 0; i < model.nv; ++i) {
            q(i) = rng.range(-0.6, 0.6);
            v(i) = rng.range(-1.2, 1.2);
        }
    }

    void row(const std::string& model, const std::string& op, double us)
    {
        std::printf("| %-30s | %-32s | %10.2f |\n", model.c_str(), op.c_str(), us);
    }

    // Task 4.8: Eigen's general sparse Cholesky on the mass matrix's lower
    // triangle, its pattern analysed once and not reordered (the tree's
    // numbering has no fill-in); timed with the copy of M's values into it.
    double sparse_factor_and_solve_us(const kernel::TreeCholesky& tc, const MatX& M)
    {
        const Index n = M.rows();
        std::vector<Eigen::Triplet<double>> entries;
        for (Index k = 0; k < n; ++k) {
            entries.emplace_back(static_cast<int>(k), static_cast<int>(k), M(k, k));
            for (Index i = tc.parent(k); i >= 0; i = tc.parent(i)) {
                entries.emplace_back(static_cast<int>(k), static_cast<int>(i), M(k, i));
            }
        }
        Eigen::SparseMatrix<double> S(n, n);
        S.setFromTriplets(entries.begin(), entries.end());
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> llt;
        llt.analyzePattern(S);
        const VecX b = VecX::Ones(n);
        VecX x = b;
        return time_per_call_us([&] {
            for (int j = 0; j < S.outerSize(); ++j) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(S, j); it; ++it) {
                    it.valueRef() = M(it.row(), it.col());
                }
            }
            llt.factorize(S);
            x = llt.solve(b);
        });
    }
}

int main()
{
    std::printf("| Model | Operation | Time [us] |\n");
    std::printf("|---|---|---|\n");

    for (bool tree : {false, true}) {
        for (int n : {2, 4, 8, 16, 32, 64}) {
            kernel::Model model;
            VecX q, v;
            // A binary tree (body i a child of body i / 2) or a chain.
            build_revolute_system(model, q, v, n, [tree](int i) { return tree ? i / 2 : i - 1; });
            model.gravity = Vec3(0.0, -g_accel, 0.0);
            kernel::Data data(model);
            const VecX zero = VecX::Zero(model.nv);
            const std::string name = std::string(tree ? "revolute tree, " : "revolute chain, ")
                                   + std::to_string(n) + " bodies";

            row(name, "positions and velocities",
                time_per_call_us([&] { kernel::forward_kinematics(model, data, q, v); }));
            row(name, "mass matrix",
                time_per_call_us([&] { volatile double x = kernel::crba(model, data, q)(0, 0); (void)x; }));
            row(name, "inverse dynamics",
                time_per_call_us([&] { volatile double x = kernel::rnea(model, data, q, v, zero)(0); (void)x; }));
            row(name, "forward dynamics",
                time_per_call_us([&] { volatile double x = kernel::aba(model, data, q, v, zero)(0); (void)x; }));
            // Task 4.8: the mass matrix factorized and solved densely and along
            // the tree, and the tree's work estimate against the dense one.
            const MatX M = kernel::crba(model, data, q);
            const VecX b = VecX::Ones(model.nv);
            VecX x = b;
            Eigen::LLT<MatX> llt(model.nv);
            kernel::TreeCholesky tree_factor(model);
            row(name, "mass matrix factor and solve, dense", time_per_call_us([&] {
                llt.compute(M); x = b; llt.solveInPlace(x); }));
            row(name, "mass matrix factor and solve, tree", time_per_call_us([&] {
                tree_factor.compute(M); x = b; tree_factor.solve(x); }));
            row(name, "mass matrix factor and solve, Eigen sparse", sparse_factor_and_solve_us(tree_factor, M));
            row(name, "tree work / dense work (ratio, not us)", tree_factor.work() / tree_factor.dense_work());
        }
    }

    // Task 4.8, the crossover: the mass matrix factorized densely and along
    // the tree, on a root body carrying 1, 2 or 4 chains numbered one after
    // the other (a chain, a fork, a car's corners), at the sizes around
    // where the two meet.
    for (int branches : {1, 2, 4}) {
        for (int n : {12, 16, 24, 32, 48, 64}) {
            const int length = (n - 1 + branches - 1) / branches;   // bodies per chain
            kernel::Model model;
            VecX q, v;
            build_revolute_system(model, q, v, n, [length](int i) {
                return i == 1 ? 0 : (i - 2) % length == 0 ? 1 : i - 1; });
            kernel::Data data(model);
            const std::string name = "root and " + std::to_string(branches) + " chain"
                                   + (branches > 1 ? "s, " : ", ") + std::to_string(n) + " bodies";
            const MatX M = kernel::crba(model, data, q);
            const VecX b = VecX::Ones(model.nv);
            VecX x = b;
            Eigen::LLT<MatX> llt(model.nv);
            kernel::TreeCholesky tree_factor(model);
            row(name, "mass matrix factor and solve, dense", time_per_call_us([&] {
                llt.compute(M); x = b; llt.solveInPlace(x); }));
            row(name, "mass matrix factor and solve, tree", time_per_call_us([&] {
                tree_factor.compute(M); x = b; tree_factor.solve(x); }));
            row(name, "mass matrix factor and solve, Eigen sparse", sparse_factor_and_solve_us(tree_factor, M));
            row(name, "tree work / dense work (ratio, not us)", tree_factor.work() / tree_factor.dense_work());
        }
    }

    // The detailed vehicle: chassis on a free joint, steering rack, four
    // double-wishbone corners closed by constraints, springs, dampers and
    // tyres, driven through the drivetrain.
    {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
        tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;
        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);
        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        Drivetrain dt(tmpl.drivetrain);
        dt.initialize(sim, vh);
        dt.connect(sim, vh);
        dt.throttle = 0.3;
        sim.run(0.2, 0.001);   // moving, wheels loaded

        const std::string name = "double-wishbone sedan";
        const kernel::Model& model = sys.model;
        kernel::Data data(model);
        kernel::ConstraintSolver solver(model, sys.constraints);
        const VecX zero = VecX::Zero(model.nv);
        VecX q = sim.q, v = sim.v;
        row(name, "positions and velocities", time_per_call_us([&] {
            kernel::forward_kinematics(model, data, q, v); }));
        row(name, "constraint equations", time_per_call_us([&] { solver.evaluate(data, q, v, 0.0); }));
        row(name, "mass matrix", time_per_call_us([&] {
            volatile double x = kernel::crba(model, data, q)(0, 0); (void)x; }));
        row(name, "bias forces", time_per_call_us([&] {
            volatile double x = kernel::rnea(model, data, q, v, zero)(0); (void)x; }));
        solver.mass_factorization = kernel::ConstraintSolver::MassFactorization::Dense;
        row(name, "constrained forward dynamics", time_per_call_us([&] {
            volatile double x = solver.forward_dynamics(data, q, v, zero, 0.0)(0); (void)x; }));
        solver.mass_factorization = kernel::ConstraintSolver::MassFactorization::Tree;
        row(name, "constrained forward dynamics, tree", time_per_call_us([&] {
            volatile double x = solver.forward_dynamics(data, q, v, zero, 0.0)(0); (void)x; }));
        row(name, "tree work / dense work (ratio, not us)",
            solver.tree_cholesky().work() / solver.tree_cholesky().dense_work());
        row(name, "velocities (count, not us)", static_cast<double>(model.nv));
        solver.mass_factorization = kernel::ConstraintSolver::MassFactorization::Dense;
        row(name, "accelerations, with the forces", time_per_call_us([&] {
            volatile double x = sim.acceleration(q, v, sim.time)(0); (void)x; }));
        row(name, "constraint projection", time_per_call_us([&] {
            VecX qp = q, vp = v;
            solver.project(data, qp, vp, 0.0); }));
        sim.solver().mass_factorization = kernel::ConstraintSolver::MassFactorization::Dense;
        row(name, "one RK4 step of 1 ms, everything", time_per_call_us([&] { sim.step(0.001); }));
        sim.solver().mass_factorization = kernel::ConstraintSolver::MassFactorization::Tree;
        row(name, "one RK4 step of 1 ms, tree", time_per_call_us([&] { sim.step(0.001); }));
        sim.solver().mass_factorization = kernel::ConstraintSolver::MassFactorization::Automatic;
    }

    // The integrators on the sedan (task 4.5): the median time of one 1 ms
    // step at the state above, and the spread of single steps over a drive
    // (throttle for 1 s, brake to a stop for 2 s, stand for 1 s), after the
    // first ten (which size the work and warm the caches): what a real time
    // budget must allow for. Generalised-alpha also solved on the constraints
    // (task 4.7's option).
    std::printf("\n| Method | median of one 1 ms step [us] | single steps: min | median | 99th percentile | 99.9th "
                "| max | evaluations per step | Newton iterations per step |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|\n");
    struct Run {
        kernel::Integrator method;
        bool stabilized;
        const char* label;
    };
    for (const Run run : {Run{kernel::Integrator::RK4, false, "RK4"},
                          Run{kernel::Integrator::GeneralizedAlpha, false, "generalised-alpha"},
                          Run{kernel::Integrator::GeneralizedAlpha, true, "generalised-alpha, stabilized"},
                          Run{kernel::Integrator::LinearlyImplicitEuler, false, "linearly implicit Euler"}}) {
        const kernel::Integrator method = run.method;
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
        tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;
        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);
        kernel::Simulator sim(sys);
        sim.method = method;
        sim.implicit_control.stabilized_index2 = run.stabilized;
        set_vehicle_equilibrium(sim, vh);
        sim.initialize();
        Drivetrain dt(tmpl.drivetrain);
        dt.initialize(sim, vh);
        dt.connect(sim, vh);
        dt.throttle = 0.3;
        sim.run(0.2, 0.001);
        const double median = time_per_call_us([&] { sim.step(0.001); });

        // A fresh run for the single steps.
        kernel::System sys2;
        auto vh2 = build_vehicle(sys2, tmpl);
        kernel::Simulator sim2(sys2);
        sim2.method = method;
        sim2.implicit_control.stabilized_index2 = run.stabilized;
        set_vehicle_equilibrium(sim2, vh2);
        sim2.initialize();
        Drivetrain dt2(tmpl.drivetrain);
        dt2.initialize(sim2, vh2);
        dt2.connect(sim2, vh2);
        std::vector<double> steps;
        steps.reserve(4000);
        for (int k = 0; k < 4000; ++k) {
            dt2.throttle = k < 1000 ? 0.5 : 0.0;
            dt2.brake = k >= 1000 && k < 3000 ? 0.6 : 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            sim2.step(0.001);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (k >= 10) steps.push_back(us);
        }
        std::sort(steps.begin(), steps.end());
        const char* label = run.label;
        const kernel::IntegratorStats& st = sim2.integrator_stats();
        std::printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f |\n", label, median, steps.front(),
                    steps[steps.size() / 2], steps[steps.size() * 99 / 100], steps[steps.size() * 999 / 1000],
                    steps.back(), static_cast<double>(st.evaluations) / static_cast<double>(st.steps),
                    static_cast<double>(st.newton_iterations) / static_cast<double>(st.steps));
    }
    return 0;
}
