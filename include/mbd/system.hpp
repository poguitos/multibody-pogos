#pragma once

#include <vector>
#include <memory>
#include "mbd/core.hpp"
#include "mbd/rigid_body.hpp"
#include "mbd/force_element.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

class Constraint; // Forward declaration

// The central container for a multibody simulation.
// Owns bodies, constraints, and forces.
class MultibodySystem {
public:
    // Body data stored as parallel arrays (Structure of Arrays)
    // for efficient access during solver loops.
    std::vector<RigidBodyState> states;
    std::vector<RigidBodyInertia> inertias;
    std::vector<RigidBodyForces> forces;
    
    // Components
    std::vector<std::unique_ptr<ForceElement>> force_elements;
    std::vector<std::shared_ptr<Constraint>> constraints;

    MultibodySystem() = default;

    // Add a rigid body to the system. Returns its unique BodyIndex.
    BodyIndex add_body(const RigidBodyInertia& inertia, 
                       const RigidBodyState& initial_state = RigidBodyState()) 
    {
        states.push_back(initial_state);
        inertias.push_back(inertia);
        forces.push_back(RigidBodyForces{}); // Init with zero forces
        return static_cast<BodyIndex>(states.size() - 1);
    }

    BodyIndex body_count() const { return static_cast<BodyIndex>(states.size()); }

    // Clear all accumulated forces (call this at the start of a time step)
    void clear_forces() {
        for (auto& f : forces) {
            f.f_W.setZero();
            f.tau_W.setZero();
        }
    }

    // Accumulate forces from all force elements
    void apply_forces() {
        for (const auto& fe : force_elements) {
            // ForceElements currently are usually 1-body or 2-body.
            // A general ForceElement might need access to the whole System 
            // to find its bodies.
            // For the simple LinearSpringDamper we implemented in C1.1.2, 
            // it accumulates into a RigidBodyForces struct. 
            // We need to map that logic here.
            
            // NOTE: To properly support ForceElements acting on specific bodies in the System,
            // ForceElement::apply needs to know which body to act on.
            // For now, we will defer updating ForceElement to keep this step focused on Constraints.
            // We can manually apply forces in tests or loops for now.
        }
    }
};

} // namespace mbd