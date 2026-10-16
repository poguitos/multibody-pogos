#pragma once

// The linearly implicit Euler step's work (simulator_linear.cpp), defined here so
// that the simulator's other files can construct and destroy it. Internal to
// the library.

#include <cstddef>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd::kernel {

struct Simulator::LinearWork {
    // What the caches below were built for.
    std::size_t force_elements{0}, joint_forces{0};
    Index nv{-1}, nx{-1};
    // Each element's bodies (bodies() allocates) and the largest count.
    std::vector<std::vector<BodyIndex>> element_bodies;
    std::vector<std::vector<int>> joint_bodies;
    int max_bodies{0};
    // For each body an element acts on, the columns of its Jacobian that can
    // be nonzero (its own joint's and its ancestors') and their values.
    std::vector<char> body_needed;
    std::vector<std::vector<Index>> body_columns;
    std::vector<MatX> body_jacobian;
    MatX jacobian_full;
    // One element's load derivatives (6 rows and columns per body), the body
    // states and loads it is evaluated with, and its loads at the base state.
    MatX load_K, load_C;
    std::vector<RigidBodyState> states;
    std::vector<RigidBodyForces> forces;
    std::vector<Vec6> load0;
    // The generalized derivatives, the iteration matrix, the accelerations at
    // the step's start and work vectors.
    MatX K, C, W;
    VecX a, r, dv, v1;
    // Joint forces: perturbed q and v, generalized forces, a coordinate step.
    VecX q1, tau0, tau1, step;
    // States: X = d x_dot / dx, Xv = d x_dot / dv, Tx = d tau / dx; I - h X
    // factorized with its pivots; G = (I - h X)^-1 h Xv; one element's
    // rates' derivatives with respect to its bodies' velocities (6 columns
    // per body); the rates and generalized forces at the step's start and
    // perturbed; perturbed states; the states' right-hand side and their
    // change.
    MatX X, Xv, Tx, Xm, G, load_X;
    std::vector<Index> X_pivots;
    VecX xd0, xd1, x1, tau_x0, tau_x1, rhs_x, dx0, dx;
    bool fallback_reported{false};
};

} // namespace mbd::kernel
