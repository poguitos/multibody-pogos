#pragma once

// Anti-roll bar (ARB): torsion spring coupling two wheels on an axle.
//
// Measures vertical displacement (in chassis frame) of each wheel center
// from reference, and applies restoring forces proportional to the
// difference. Opposes roll without affecting symmetric bounce.

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

class AntiRollBar : public ForceElement {
public:
    BodyIndex chassis_idx;
    BodyIndex left_wheel_idx;
    BodyIndex right_wheel_idx;

    /// Roll stiffness [N/m] per unit travel difference.
    /// Typical values: 20,000-80,000 N/m for passenger cars.
    Real k_arb{30000.0};

    /// Roll damping [Ns/m] per unit travel difference velocity.
    Real c_arb{0.0};

    /// Reference Y coordinate of left wheel in chassis frame.
    /// Usually set automatically from initial configuration.
    Real z_L_ref{0.0};

    /// Reference Y coordinate of right wheel in chassis frame.
    Real z_R_ref{0.0};

    AntiRollBar(BodyIndex chassis,
                BodyIndex left_wheel,
                BodyIndex right_wheel,
                Real stiffness,
                Real damping = 0.0);

    /// Capture reference heights from current state.
    /// Call this ONCE after the vehicle is built and positioned at equilibrium,
    /// before starting the simulation.
    void capture_reference(const std::vector<RigidBodyState>& states);

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;
};

} // namespace mbd
