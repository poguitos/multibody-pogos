#pragma once

// Contact of points and spheres with a plane (plan task 3.8, decision D28).
//
// A sphere of radius r (r = 0: a point) centred at c on one body touches the
// plane through the origin p of a marker on another body (the ground, body
// 0, or any other), normal to the marker's Z axis n, when its penetration
//
//     delta = r - n . (c - p)
//
// is positive. The normal force pushes the sphere out along n:
//
//     F_n = k delta^e + c(delta) delta_dot,   never below zero,
//
// with delta_dot the rate of penetration. The damping c(delta) rises from
// zero at the surface to c_max at the depth d_max along a cubic step, so the
// force is continuous where contact begins and ends: a damper at full
// strength from the first touch would jump on landing and pull the bodies
// together as they part. This is the IMPACT function of ADAMS.
//
// Friction opposes the sliding velocity v_t of the contact point (the
// sphere's lowest point c - r n, on its body, relative to the plane's body,
// in the plane): F_t = -mu(|v_t|) F_n v_t / |v_t|. The coefficient is
// regularised: it rises from 0 at rest to mu_static at v_static along a cubic
// step, falls along another to mu_dynamic at v_dynamic, and stays there.
// Nothing sticks exactly: below v_static the contact creeps at the speed at
// which mu(v) F_n balances what pushes it along, slower the smaller
// v_static.

#include <vector>

#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

struct ContactParams {
    Real stiffness{0.0};          ///< k [N/m^e]
    Real exponent{1.0};           ///< e >= 1 (1.5 for Hertzian contact of a sphere)
    Real damping{0.0};            ///< c_max [N s/m]
    Real damping_depth{1e-4};     ///< d_max [m], > 0
    Real mu_static{0.0};
    Real mu_dynamic{0.0};
    Real v_static{1e-4};          ///< [m/s], > 0
    Real v_dynamic{1e-3};         ///< [m/s], > v_static
};

class PlaneContact : public ForceElement {
public:
    /// Spheres of radius `radius` centred at `centres_B` on body `body` (in
    /// its frame), against the XY plane of the frame `plane_X` on body
    /// `plane_body`, whose Z axis points out of the plane's solid side.
    PlaneContact(BodyIndex body, std::vector<Vec3> centres_B, Real radius,
                 BodyIndex plane_body, const Transform3& plane_X, ContactParams params);

    const char* name() const override { return "plane contact"; }
    std::vector<BodyIndex> bodies() const override { return {body_, plane_body_}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// The normal force at penetration delta and rate delta_dot, with its
    /// derivatives with respect to both.
    LawValue normal_law(Real delta, Real delta_rate) const;

    /// The friction coefficient at sliding speed v, in `force`, with its
    /// derivative with respect to v in `d_rate`.
    LawValue friction(Real speed) const;

    /// The penetrations of the spheres at the given states (negative: apart).
    std::vector<Real> penetrations(const std::vector<RigidBodyState>& states) const;

    const ContactParams& params() const { return params_; }

private:
    BodyIndex body_, plane_body_;
    std::vector<Vec3> centres_B_;
    Real radius_;
    Transform3 plane_X_;
    ContactParams params_;
};

} // namespace mbd
