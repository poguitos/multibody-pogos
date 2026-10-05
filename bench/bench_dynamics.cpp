// Timing of the dynamics algorithms (plan task 2.10).
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

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/simulator.hpp"
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

    // n bodies on revolute joints with generic axes, and a random state.
    // `tree` makes body i a child of body i / 2 (a binary tree); otherwise
    // each body is a child of the previous one.
    void build_revolute_system(kernel::Model& model, VecX& q, VecX& v, int n, bool tree)
    {
        Lcg rng;
        const auto revolute = std::make_shared<kernel::RevoluteJointModel>();
        for (int i = 1; i <= n; ++i) {
            const BodyIndex parent = tree ? BodyIndex(i / 2) : BodyIndex(i - 1);
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
}

int main()
{
    std::printf("| Model | Operation | Time [us] |\n");
    std::printf("|---|---|---|\n");

    for (bool tree : {false, true}) {
        for (int n : {2, 4, 8, 16, 32, 64}) {
            kernel::Model model;
            VecX q, v;
            build_revolute_system(model, q, v, n, tree);
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
        row(name, "constrained forward dynamics", time_per_call_us([&] {
            volatile double x = solver.forward_dynamics(data, q, v, zero, 0.0)(0); (void)x; }));
        row(name, "accelerations, with the forces", time_per_call_us([&] {
            volatile double x = sim.acceleration(q, v, sim.time)(0); (void)x; }));
        row(name, "constraint projection", time_per_call_us([&] {
            VecX qp = q, vp = v;
            solver.project(data, qp, vp, 0.0); }));
        row(name, "one RK4 step of 1 ms, everything", time_per_call_us([&] { sim.step(0.001); }));
    }
    return 0;
}
