#pragma once

// The two exporters of mbd_crosscheck (plan task 5.3) and what they share.

#include <cstddef>
#include <vector>

#include "mbd/kernel/model.hpp"

#include "kernel/kernel_helpers.hpp"
#include "support/rng.hpp"

namespace mbd_crosscheck {

/// Random trees and states with this library's ABA, RNEA, CRBA and
/// placements, for compare_pinocchio.py.
int export_trees(const char* path, int trees, unsigned long long seed);

/// Random closed mechanisms with this library's accelerations and loads at
/// rest and its motion from rest, for compare_chrono.py.
int export_mechanisms(const char* path, int mechanisms, unsigned long long seed);

/// A random tree of n bodies, each hung from a randomly chosen earlier body
/// (or the ground) on a random joint of any kind.
inline mbd::kernel::Model random_branching_tree(mbd_test::Rng& rng, int n)
{
    const std::vector<mbd_test::KJoint>& kinds = mbd_test::all_kernel_joints();
    mbd::kernel::Model model;
    for (int b = 1; b <= n; ++b) {
        const int parent = static_cast<int>(rng.range(0.0, static_cast<mbd::Real>(b) - 1e-9));
        const mbd_test::KJoint kind =
            kinds[static_cast<std::size_t>(rng.range(0.0, static_cast<mbd::Real>(kinds.size()) - 1e-9))];
        mbd_test::add_random_body(model, rng, kind, parent);
    }
    return model;
}

} // namespace mbd_crosscheck
