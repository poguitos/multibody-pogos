#pragma once

// Forces on joint coordinates (plan task 3.1, decision D24): elements that add
// to the generalized forces directly, from the joint's own coordinate and
// rate. Exact and cheap for what acts across a joint: a torsion spring in a
// hinge, friction in a slider, a limit stop.

#include <limits>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel {

/// Interface of the elements that act on generalized forces directly.
class JointForce {
public:
    virtual ~JointForce() = default;

    /// Short name, for messages.
    virtual const char* name() const = 0;

    /// The bodies whose joints the element acts on, for the checks.
    virtual std::vector<int> bodies() const = 0;

    /// Add the element's generalized forces at (q, v) to tau.
    virtual void apply(const VecX& q, const VecX& v, VecX& tau) const = 0;

protected:
    JointForce() = default;
};

struct JointCoordinateForceParams {
    /// Coordinate at which the spring carries no load [rad or m].
    Real reference{0.0};
    /// Restoring force or torque against the deflection q - reference.
    Curve spring;
    /// Constant force or torque in the direction of increasing q.
    Real preload{0.0};
    /// Force or torque opposing the rate v; positive for a positive rate.
    Curve damper;
    /// Coulomb friction level, opposing the motion [N or N m]. Regularised:
    /// friction * tanh(v / friction_velocity), so it builds up over rates of
    /// the order of friction_velocity, and a slow creep replaces sticking.
    Real friction{0.0};
    Real friction_velocity{1e-3};
    /// Limit stops: beyond a limit, a spring and damper push back, and never
    /// pull.
    Real lower_limit{-std::numeric_limits<Real>::infinity()};
    Real upper_limit{std::numeric_limits<Real>::infinity()};
    Real limit_stiffness{0.0};
    Real limit_damping{0.0};
};

/// Force or torque on the coordinate of a joint with one coordinate (revolute
/// or prismatic), given by the joint's child body: spring, preload, damper,
/// friction and limit stops.
class JointCoordinateForce final : public JointForce {
public:
    JointCoordinateForce(const Model& model, int body, JointCoordinateForceParams params);

    const char* name() const override { return "joint coordinate force"; }
    std::vector<int> bodies() const override { return {body_}; }
    void apply(const VecX& q, const VecX& v, VecX& tau) const override;

    /// The generalized force at coordinate q and rate v, with its derivatives.
    LawValue law(Real q, Real v) const;

    /// Energy stored at coordinate q by the spring, the preload and the stop
    /// springs, zero at the reference: the conservative part of the force is
    /// -dV/dq. Damping and friction are not conservative.
    Real potential_energy(Real q) const;

    const JointCoordinateForceParams& params() const { return params_; }

private:
    int body_;
    int iq_, iv_;
    JointCoordinateForceParams params_;
};

} // namespace mbd::kernel
