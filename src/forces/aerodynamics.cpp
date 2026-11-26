#include "mbd/forces/aerodynamics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "mbd/forces/road.hpp"

namespace mbd {

// ============================================================================
// The map
// ============================================================================

AeroMap AeroMap::constant(const Vec6& coefficients)
{
    AeroMap m;
    m.values = {coefficients};
    return m;
}

AeroMap AeroMap::from(const AeroParams& p)
{
    if (p.dClA_dh == 0.0) return constant(p.constant_coefficients(p.h_ref));
    // The constant model is linear in the height below h_ref and constant
    // above it: two nodes reproduce it from 10 m below h_ref upwards.
    AeroMap m;
    m.nodes[0] = {p.h_ref - 10.0, p.h_ref};
    m.values = {p.constant_coefficients(p.h_ref - 10.0), p.constant_coefficients(p.h_ref)};
    return m;
}

void AeroMap::validate() const
{
    std::size_t count = 1;
    for (const std::vector<Real>& g : nodes) {
        MBD_THROW_IF(g.empty(), "MBD-F060: AeroMap: a variable has no nodes");
        for (std::size_t i = 0; i < g.size(); ++i) {
            MBD_THROW_IF(!std::isfinite(g[i]) || (i > 0 && !(g[i] > g[i - 1])),
                         "MBD-F060: AeroMap: a variable's nodes are not finite and strictly increasing");
        }
        count *= g.size();
    }
    MBD_THROW_IF(values.size() != count,
                 "MBD-F060: AeroMap: the number of coefficient sets is not the product of the variables' numbers of nodes");
    for (const Vec6& v : values) {
        MBD_THROW_IF(!v.allFinite(), "MBD-F060: AeroMap: a coefficient is not finite");
    }
}

Vec6 AeroMap::at(const std::array<Real, 5>& x) const
{
    // Per variable: the node below x and the weight of the node above it;
    // held (weight 0) at or beyond the ends.
    std::array<std::size_t, 5> below{};
    std::array<Real, 5> weight{};
    std::array<std::size_t, 5> stride{};
    std::size_t s = 1;
    for (std::size_t a = 0; a < 5; ++a) {
        const std::vector<Real>& g = nodes[a];
        stride[a] = s;
        s *= g.size();
        if (g.size() == 1 || x[a] <= g.front()) {
            below[a] = 0;
        } else if (x[a] >= g.back()) {
            below[a] = g.size() - 1;
        } else {
            below[a] = static_cast<std::size_t>(std::upper_bound(g.begin(), g.end(), x[a]) - g.begin()) - 1;
            weight[a] = (x[a] - g[below[a]]) / (g[below[a] + 1] - g[below[a]]);
        }
    }
    Vec6 out = Vec6::Zero();
    for (unsigned corner = 0; corner < 32; ++corner) {
        Real w = 1.0;
        std::size_t index = 0;
        bool used = true;
        for (std::size_t a = 0; a < 5 && used; ++a) {
            const bool up = (corner >> a) & 1u;
            if (up && weight[a] == 0.0) used = false;   // a node of zero weight (or none above)
            w *= up ? weight[a] : 1.0 - weight[a];
            index += (below[a] + (up ? 1 : 0)) * stride[a];
        }
        if (used) out += w * values[index];
    }
    return out;
}

// ============================================================================
// The constant model
// ============================================================================

Vec6 AeroParams::constant_coefficients(Real h) const
{
    Real ClA_eff = ClA;
    if (dClA_dh != 0.0 && h < h_ref) ClA_eff += dClA_dh * (h_ref - h);
    const Vec3 A(-CdA, 0.0, -ClA_eff);
    Vec6 c;
    c.head<3>() = A;
    c.tail<3>() = (cop_offset_chassis - reference_point).cross(A);   // about the reference point
    return c;
}

// ============================================================================
// The force element
// ============================================================================

AerodynamicForce::AerodynamicForce(BodyIndex chassis, const AeroParams& p)
    : chassis_idx(chassis), params(p)
{
    MBD_THROW_IF(p.CdA < 0.0, "MBD-F020: AerodynamicForce: CdA must be non-negative");
    MBD_THROW_IF(p.ClA < 0.0, "MBD-F020: AerodynamicForce: ClA must be non-negative");
    MBD_THROW_IF(p.dClA_dh < 0.0, "MBD-F020: AerodynamicForce: dClA_dh must be non-negative");
    MBD_THROW_IF(p.air_density <= 0.0, "MBD-F020: AerodynamicForce: air_density must be positive");
    if (p.map) p.map->validate();
}

AerodynamicForce::Inputs AerodynamicForce::inputs(const RigidBodyState& s) const
{
    const Vec3& n = params.ground_normal_W;
    const Vec3 xB = s.q_WB * Vec3::UnitX();
    const Vec3 yB = s.q_WB * Vec3::UnitY();
    const Vec3 zB = s.q_WB * Vec3::UnitZ();
    auto height = [&](const Vec3& point) {
        const Vec3 p = s.p_WB + s.q_WB * point;
        return params.road ? p.z() - params.road->at(p.x(), p.y()).height : n.dot(p);
    };
    Inputs in;
    in.x[0] = height(params.front_point);
    in.x[1] = height(params.rear_point);
    in.x[2] = std::asin(std::clamp(-xB.dot(n), Real(-1.0), Real(1.0)));
    in.x[3] = std::atan2(yB.dot(n), zB.dot(n));
    // The reference point's velocity relative to the air, along the ground.
    const Vec3 r = s.q_WB * params.reference_point;
    Vec3 v = s.v_WB + s.w_WB.cross(r);
    if (params.wind) v -= params.wind(s.p_WB + r);
    v -= n.dot(v) * n;
    in.x[4] = v.squaredNorm() > 0.0 ? std::atan2(v.dot(yB), v.dot(xB)) : 0.0;
    in.q = 0.5 * params.air_density * v.squaredNorm();
    return in;
}

void AerodynamicForce::apply(const std::vector<RigidBodyState>& states,
                             std::vector<RigidBodyForces>& forces) const
{
    const RigidBodyState& s = states[chassis_idx];
    const Inputs in = inputs(s);
    if (in.q < 0.5 * params.air_density * 1e-12) return;   // below 1e-6 m/s: none

    const Vec6 c = params.map ? params.map->at(in.x) : params.constant_coefficients(in.x[0]);
    const Vec3 F_W = s.q_WB * (in.q * c.head<3>());
    const Vec3 M_W = s.q_WB * (in.q * c.tail<3>());   // about the reference point
    const Vec3 r = s.q_WB * params.reference_point;
    forces[chassis_idx].f_W += F_W;
    forces[chassis_idx].tau_W += M_W + r.cross(F_W);
}

} // namespace mbd
