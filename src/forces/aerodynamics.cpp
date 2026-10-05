#include "mbd/forces/aerodynamics.hpp"

namespace mbd {

AerodynamicForce::AerodynamicForce(BodyIndex chassis, const AeroParams& p)
    : chassis_idx(chassis), params(p)
{
    MBD_THROW_IF(p.CdA < 0.0, "MBD-F020: AerodynamicForce: CdA must be non-negative");
    MBD_THROW_IF(p.ClA < 0.0, "MBD-F020: AerodynamicForce: ClA must be non-negative");
    MBD_THROW_IF(p.air_density <= 0.0, "MBD-F020: AerodynamicForce: air_density must be positive");
}

void AerodynamicForce::apply(const std::vector<RigidBodyState>& states,
                             std::vector<RigidBodyForces>& forces) const
{
    const auto& chassis = states[chassis_idx];

    // Horizontal velocity of chassis CG (zero out vertical component)
    Vec3 v_horiz = chassis.v_WB;
    v_horiz.y() = 0.0;
    const Real V = v_horiz.norm();

    if (V < Real(1e-6)) return; // No aero at zero speed

    // Dynamic pressure
    const Real q_dyn = Real(0.5) * params.air_density * V * V;

    // --- Drag ---
    // F_drag = -q * CdA * v_unit (along -velocity)
    const Vec3 v_unit = v_horiz / V;
    const Vec3 F_drag = -q_dyn * params.CdA * v_unit;

    // --- Downforce ---
    // Effective ClA based on ride height
    const Real h = chassis.p_WB.y();
    Real ClA_eff = params.ClA;
    if (params.dClA_dh != Real(0.0) && h < params.h_ref) {
        ClA_eff += params.dClA_dh * (params.h_ref - h);
    }
    ClA_eff = std::max(ClA_eff, Real(0.0));

    const Vec3 F_lift = Vec3(0.0, -q_dyn * ClA_eff, 0.0); // -Y direction

    // --- Total force at CoP ---
    const Vec3 F_total = F_drag + F_lift;

    // CoP world position (relative to chassis CG)
    const Vec3 r_cop_W = chassis.q_WB * params.cop_offset_chassis;

    // Apply force at CoP and corresponding moment about chassis CG
    forces[chassis_idx].f_W   += F_total;
    forces[chassis_idx].tau_W += r_cop_W.cross(F_total);
}

} // namespace mbd
