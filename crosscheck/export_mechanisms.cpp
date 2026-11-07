// Random closed mechanisms for the comparison with Project Chrono (plan task
// 5.3).
//
// Writes, as JSON, random trees closed by random loop constraints, each at a
// configuration where its loops are assembled, with this library's answers:
//
// - at rest there: every body's angular acceleration and the acceleration of
//   its centre of mass, the constraint wrench on each body (the sum over the
//   closures) and each tree joint's reaction;
// - the motion from rest under gravity for kDuration, every body's placement
//   at kSamples equal intervals, by RK4 at kStep with projection onto the
//   constraints, with the largest change from a run at twice the step as an
//   estimate of its error.
//
// compare_chrono.py builds the same mechanisms in PyChrono and compares.
//
// Each mechanism is a tree of 3 to 8 bodies on random joints of every kind,
// made by the tests' generator (tests/kernel/kernel_helpers.hpp), closed by
// one to three constraints of the kinds Chrono has a link for: spherical,
// revolute, universal, cylindrical and prismatic closures, and distance. A
// closure joins a random body to another or to the ground; its markers
// coincide at the configuration drawn. A draw whose equations are redundant
// or leave no motion is drawn again: Chrono's direct solvers need
// independent equations, and a mechanism that cannot move compares nothing.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/validate.hpp"

#include "export.hpp"
#include "json_out.hpp"

using mbd::Real;
using mbd::Transform3;
using mbd::Vec3;
using mbd::Vec6;
using mbd::VecX;
using mbd::kernel::Data;
using mbd::kernel::Marker;
using mbd::kernel::Model;
using mbd::kernel::Simulator;
using mbd::kernel::System;
using mbd_test::Rng;

namespace mbd_crosscheck {
namespace {

constexpr Real kDuration = 0.5;   // [s] of motion from rest
constexpr int kSamples = 5;       // placements written every kDuration / kSamples
constexpr Real kStep = 1e-4;      // [s] RK4 step; the error estimate compares twice this

const char* const kClosureKinds[] = {"spherical", "revolute", "universal", "cylindrical", "prismatic", "distance"};

struct Closure {
    std::string kind;
    Marker i, j;
    Real length{0.0};   // the distance's
};

std::shared_ptr<const mbd::kernel::ConstraintModel> make_constraint(const Closure& c)
{
    using namespace mbd::kernel;
    if (c.kind == "spherical") return spherical_closure(c.i, c.j);
    if (c.kind == "revolute") return revolute_closure(c.i, c.j);
    if (c.kind == "universal") return universal_closure(c.i, c.j);
    if (c.kind == "cylindrical") return cylindrical_closure(c.i, c.j);
    if (c.kind == "prismatic") return prismatic_closure(c.i, c.j);
    return std::make_shared<Distance>(c.i, c.j, c.length);
}

/// A closure from a random body to any other body or the ground, assembled
/// at the placements in `data`: marker j coincides with marker i, except
/// that for the universal it is turned a quarter turn about X (the Z axes,
/// the cross's arms, perpendicular) and for the distance it is moved by a
/// random offset of at least 0.1 m, the length taken there.
Closure random_closure(Rng& rng, const Model& model, const Data& data)
{
    const int n = model.nbodies() - 1;
    Closure c;
    c.kind = kClosureKinds[static_cast<std::size_t>(rng.range(0.0, 6.0 - 1e-9))];
    const int a = 1 + static_cast<int>(rng.range(0.0, static_cast<Real>(n) - 1e-9));
    int b = static_cast<int>(rng.range(0.0, static_cast<Real>(n) - 1e-9));
    if (b >= a) ++b;
    c.i = Marker{a, rng.frame(0.3)};
    const Transform3 Wi = data.oMi[static_cast<std::size_t>(a)] * c.i.X_BM;
    Transform3 Wj = Wi;
    if (c.kind == "universal") {
        Wj = Wi * Transform3(mbd::Quat(Eigen::AngleAxisd(0.5 * mbd::pi, Vec3::UnitX())), Vec3::Zero());
    } else if (c.kind == "distance") {
        Vec3 offset = rng.vec(0.5);
        while (offset.norm() < 0.1) offset = rng.vec(0.5);
        Wj = Transform3(Wi.q, Wi.p + offset);
        c.length = offset.norm();
    }
    c.j = Marker{b, data.oMi[static_cast<std::size_t>(b)].inverse() * Wj};
    return c;
}

std::string marker(const Marker& m)
{
    return "{\"body\": " + std::to_string(m.body) + ", \"X_BM\": " + transform(m.X_BM) + "}";
}

/// Every body's placement at kSamples equal intervals of the motion from
/// rest at q0, by RK4 at step dt.
std::vector<std::vector<Transform3>> motion(System& sys, const VecX& q0, Real dt)
{
    const Model& model = sys.model;
    Simulator sim(sys);
    sim.method = mbd::kernel::Integrator::RK4;
    sim.q = q0;
    sim.v = VecX::Zero(model.nv);
    sim.initialize();
    Data data(model);
    std::vector<std::vector<Transform3>> poses;
    for (int k = 0; k < kSamples; ++k) {
        sim.run(kDuration / kSamples, dt);
        mbd::kernel::forward_kinematics(model, data, sim.q);
        poses.emplace_back(data.oMi.begin() + 1, data.oMi.end());
    }
    return poses;
}

} // namespace

int export_mechanisms(const char* path, int mechanisms, unsigned long long seed)
{
    std::ofstream out(path);
    if (!out) {
        std::cerr << "cannot write " << path << "\n";
        return 1;
    }
    out << "{\"program\": \"mbd_crosscheck mechanisms\", \"seed\": " << seed << ", \"duration\": " << num(kDuration)
        << ", \"samples\": " << kSamples << ", \"step\": " << num(kStep) << ", \"mechanisms\": [\n";
    int total_draws = 0;
    for (int m = 0; m < mechanisms; ++m) {
        Rng rng(seed * 1000 + 500 + static_cast<unsigned long long>(m));
        std::unique_ptr<System> sys;
        std::vector<Closure> closures;
        VecX q;
        mbd::kernel::ValidationReport report;
        for (;;) {
            ++total_draws;
            sys = std::make_unique<System>();
            const int n = 3 + static_cast<int>(rng.range(0.0, 6.0 - 1e-9));   // 3 to 8 bodies
            sys->model = random_branching_tree(rng, n);
            sys->model.gravity = rng.vec(1.0).normalized() * mbd::g_accel;
            q = mbd_test::random_configuration(sys->model, rng);
            Data data(sys->model);
            mbd::kernel::forward_kinematics(sys->model, data, q);
            closures.clear();
            const int k = 1 + static_cast<int>(rng.range(0.0, 3.0 - 1e-9));   // 1 to 3 closures
            for (int c = 0; c < k; ++c) {
                closures.push_back(random_closure(rng, sys->model, data));
                sys->constraints.push_back(make_constraint(closures.back()));
            }
            report = mbd::kernel::validate(*sys, q);
            if (report.ok() && report.redundant_equations == 0 && report.degrees_of_freedom >= 1) break;
        }
        const Model& model = sys->model;

        // At rest: accelerations and loads.
        Simulator sim(*sys);
        sim.q = q;
        sim.v = VecX::Zero(model.nv);
        sim.initialize();
        const VecX q0 = sim.q;
        const mbd::kernel::Loads loads = mbd::kernel::compute_loads(sim);
        Data data(model);
        mbd::kernel::forward_kinematics(model, data, q0, VecX::Zero(model.nv), loads.v_dot);

        // The motion from rest, and the change from twice the step.
        const auto fine = motion(*sys, q0, kStep);
        const auto coarse = motion(*sys, q0, 2.0 * kStep);
        Real change = 0.0;
        for (std::size_t k = 0; k < fine.size(); ++k) {
            for (std::size_t b = 0; b < fine[k].size(); ++b) {
                change = std::max({change, (fine[k][b].p - coarse[k][b].p).norm(),
                                   fine[k][b].q.angularDistance(coarse[k][b].q)});
            }
        }

        out << (m ? ",\n" : "") << "{\"mechanism\": " << m << ", \"nq\": " << model.nq << ", \"nv\": " << model.nv
            << ", \"equations\": " << report.constraint_equations << ", \"dof\": " << report.degrees_of_freedom
            << ", \"gravity\": " << array(model.gravity) << ",\n \"bodies\": [";
        for (int b = 1; b < model.nbodies(); ++b) out << (b > 1 ? ",\n  " : "\n  ") << body(model, b);
        out << "],\n \"closures\": [";
        for (std::size_t c = 0; c < closures.size(); ++c) {
            out << (c ? ",\n  " : "\n  ") << "{\"kind\": \"" << closures[c].kind << "\", \"i\": " << marker(closures[c].i)
                << ", \"j\": " << marker(closures[c].j) << ", \"length\": " << num(closures[c].length) << "}";
        }
        out << "],\n \"q\": " << array(q0) << ",\n \"oMi\": [";
        for (int b = 1; b < model.nbodies(); ++b)
            out << (b > 1 ? ", " : "") << transform(data.oMi[static_cast<std::size_t>(b)]);
        out << "],\n \"at_rest\": {\"alpha\": [";
        for (int b = 1; b < model.nbodies(); ++b) {
            const Vec6 acc = mbd::kernel::body_acceleration_world(data, b);   // [angular; origin]
            out << (b > 1 ? ", " : "") << array(Vec3(acc.head<3>()));
        }
        out << "],\n  \"a_com\": [";
        for (int b = 1; b < model.nbodies(); ++b) {
            const auto i = static_cast<std::size_t>(b);
            const Vec6 acc = mbd::kernel::body_acceleration_world(data, b);
            const Vec3 r = data.oMi[i].q * model.inertia[i].com_B;   // at rest: no centripetal term
            out << (b > 1 ? ", " : "") << array(Vec3(acc.tail<3>() + acc.head<3>().cross(r)));
        }
        out << "],\n  \"constraint_W\": [";
        for (int b = 1; b < model.nbodies(); ++b)
            out << (b > 1 ? ", " : "") << array(loads.constraint_W[static_cast<std::size_t>(b)]);
        out << "],\n  \"joint_reaction\": [";
        for (int b = 1; b < model.nbodies(); ++b)
            out << (b > 1 ? ", " : "") << array(loads.joint_reaction[static_cast<std::size_t>(b)]);
        out << "]},\n \"motion\": {\"rk4_change\": " << num(change) << ", \"oMi\": [";
        for (std::size_t k = 0; k < fine.size(); ++k) {
            out << (k ? ",\n  " : "\n  ") << "[";
            for (std::size_t b = 0; b < fine[k].size(); ++b) out << (b ? ", " : "") << transform(fine[k][b]);
            out << "]";
        }
        out << "]}}";
    }
    out << "\n]}\n";
    std::cout << "wrote " << mechanisms << " mechanisms (" << total_draws << " draws) to " << path << "\n";
    return out ? 0 : 1;
}

} // namespace mbd_crosscheck
