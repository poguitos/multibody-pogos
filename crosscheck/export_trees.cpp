// Random trees for the comparison with Pinocchio (plan task 5.3).
//
// Writes, as JSON, random trees of every joint type with their joint frames
// and inertias, and for each tree a few random states with this program's
// answers: the accelerations of the articulated-body algorithm, the
// generalized forces of recursive Newton-Euler, the mass matrix of the
// composite-rigid-body algorithm, and every body's placement in the world.
// compare_pinocchio.py builds the same trees in Pinocchio and compares.
//
// The trees are made by the tests' own generator (tests/kernel/
// kernel_helpers.hpp): generic joint frames, tilted inertias, offset centres
// of mass; each new body hangs from a randomly chosen earlier one, so the
// trees branch at random depths. Gravity points in a random direction.

#include <fstream>
#include <iostream>

#include "mbd/kernel/algorithms.hpp"

#include "export.hpp"
#include "json_out.hpp"
#include "support/rng.hpp"

using mbd::VecX;
using mbd::kernel::Data;
using mbd::kernel::Model;
using mbd_test::Rng;

namespace mbd_crosscheck {

int export_trees(const char* path, int trees, unsigned long long seed)
{
    constexpr int kStates = 3;
    std::ofstream out(path);
    if (!out) {
        std::cerr << "cannot write " << path << "\n";
        return 1;
    }
    out << "{\"program\": \"mbd_crosscheck trees\", \"seed\": " << seed << ", \"trees\": [\n";
    for (int t = 0; t < trees; ++t) {
        Rng rng(seed * 1000 + static_cast<unsigned long long>(t));
        const int n = 2 + static_cast<int>(rng.range(0.0, 23.0 - 1e-9));   // 2 to 24 bodies
        Model model = random_branching_tree(rng, n);
        model.gravity = rng.vec(1.0).normalized() * mbd::g_accel;
        Data data(model);

        out << (t ? ",\n" : "") << "{\"tree\": " << t << ", \"nq\": " << model.nq << ", \"nv\": " << model.nv
            << ", \"gravity\": " << array(model.gravity) << ",\n \"bodies\": [";
        for (int b = 1; b < model.nbodies(); ++b) out << (b > 1 ? ",\n  " : "\n  ") << body(model, b);
        out << "],\n \"states\": [";
        for (int s = 0; s < kStates; ++s) {
            const VecX q = mbd_test::random_configuration(model, rng);
            const VecX v = mbd_test::random_vector(model.nv, rng, 2.0);
            const VecX tau = mbd_test::random_vector(model.nv, rng, 5.0);
            const VecX a = mbd_test::random_vector(model.nv, rng, 3.0);
            const VecX ddq = mbd::kernel::aba(model, data, q, v, tau);
            const VecX tau_rnea = mbd::kernel::rnea(model, data, q, v, a);
            const mbd::MatX M = mbd::kernel::crba(model, data, q);
            mbd::kernel::forward_kinematics(model, data, q);
            out << (s ? ",\n  " : "\n  ") << "{\"q\": " << array(q) << ", \"v\": " << array(v)
                << ", \"tau\": " << array(tau) << ", \"a\": " << array(a) << ",\n   \"ddq\": " << array(ddq)
                << ",\n   \"tau_rnea\": " << array(tau_rnea) << ",\n   \"M\": " << matrix(M) << ",\n   \"oMi\": [";
            for (int b = 1; b < model.nbodies(); ++b)
                out << (b > 1 ? ", " : "") << transform(data.oMi[static_cast<std::size_t>(b)]);
            out << "]}";
        }
        out << "]}";
    }
    out << "\n]}\n";
    std::cout << "wrote " << trees << " trees, " << kStates << " states each, to " << path << "\n";
    return out ? 0 : 1;
}

} // namespace mbd_crosscheck
