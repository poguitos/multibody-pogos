#pragma once

// Six-axis bushing between two bodies (plan task 3.1): a force and a moment
// against the relative displacement and rotation of two markers, axis by axis,
// each with its own characteristic, linear or tabulated. Its coordinates, in
// the axes of marker 1: the rotation vector of marker 2 relative to marker 1
// and the position of marker 2's origin, ordered [rotation; displacement] as
// spatial vectors are. Their rates: the relative angular velocity (the rotation
// vector's rate to first order) and the displacement's rate.

#include <array>
#include <vector>

#include "mbd/core/math.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

struct BushingParams {
    /// Moment about each axis of marker 1 (X, Y, Z) against the rotation
    /// about it, and force along each axis against the displacement.
    std::array<Curve, 3> rotational_stiffness;
    std::array<Curve, 3> stiffness;
    /// The same against the rates.
    std::array<Curve, 3> rotational_damping;
    std::array<Curve, 3> damping;
};

/// A bushing's load at given coordinates and rates: [moment; force] on body 2
/// in marker 1's axes, with its derivatives (diagonal: each axis on its own).
struct BushingLaw {
    Vec6 load{Vec6::Zero()};
    Mat6 d_position{Mat6::Zero()};
    Mat6 d_rate{Mat6::Zero()};
};

class Bushing : public ForceElement {
public:
    /// Between marker X_1M on body b1 and marker X_2M on body b2, each in its
    /// body's frame. At rest the markers coincide.
    Bushing(BodyIndex b1, const Transform3& X_1M, BodyIndex b2, const Transform3& X_2M,
            BushingParams params);

    const char* name() const override { return "bushing"; }
    std::vector<BodyIndex> bodies() const override { return {body1_, body2_}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// The coordinates and rates, in marker 1's axes.
    void deflection(const std::vector<RigidBodyState>& states, Vec6& x, Vec6& x_dot) const;

    BushingLaw law(const Vec6& x, const Vec6& x_dot) const;

private:
    BodyIndex body1_, body2_;
    Transform3 X_1M_, X_2M_;
    BushingParams params_;
};

} // namespace mbd
