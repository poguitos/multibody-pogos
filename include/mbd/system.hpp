#pragma once

#include <vector>
#include <memory>
#include <string>

#include "mbd/core.hpp"
#include "mbd/rigid_body.hpp"
#include "mbd/force_element.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

class Constraint; // Forward declaration

/// Per-body metadata (name, topology).
struct BodyInfo {
    std::string name;
    BodyIndex parent_idx{kNoParent};

    BodyInfo() = default;
    explicit BodyInfo(std::string n, BodyIndex parent = kNoParent)
        : name(std::move(n)), parent_idx(parent) {}
};

/// The central container for a multibody simulation.
///
/// Body 0 is always ground: identity pose, zero velocity, never integrated.
class MultibodySystem {
public:
    std::vector<RigidBodyState>   states;
    std::vector<RigidBodyInertia> inertias;
    std::vector<RigidBodyForces>  forces;
    std::vector<BodyInfo>         body_infos;

    std::vector<std::unique_ptr<ForceElement>>  force_elements;
    std::vector<std::shared_ptr<Constraint>>    constraints;

    /// Constructor: automatically creates ground body at index 0.
    MultibodySystem()
    {
        states.push_back(RigidBodyState{});
        inertias.push_back(RigidBodyInertia{});
        forces.push_back(RigidBodyForces{});
        body_infos.push_back(BodyInfo{"ground", kNoParent});
    }

    /// Add a body. Returns its BodyIndex (always >= 1).
    BodyIndex add_body(const RigidBodyInertia& inertia,
                       const RigidBodyState& initial_state = RigidBodyState{},
                       const std::string& name = "",
                       BodyIndex parent = kGroundIndex)
    {
        states.push_back(initial_state);
        inertias.push_back(inertia);
        forces.push_back(RigidBodyForces{});
        body_infos.push_back(BodyInfo{name, parent});
        return static_cast<BodyIndex>(states.size() - 1);
    }

    BodyIndex body_count() const { return static_cast<BodyIndex>(states.size()); }

    bool is_ground(BodyIndex idx) const { return idx == kGroundIndex; }

    void clear_forces()
    {
        for (auto& f : forces) {
            f.f_W.setZero();
            f.tau_W.setZero();
        }
    }

    void apply_force_elements()
    {
        for (const auto& fe : force_elements) {
            fe->apply(states, forces);
        }
    }
};

} // namespace mbd