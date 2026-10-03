#pragma once

// Joint models of the kinematics kernel (plan Phase 2, docs/kernel.md).
//
// A joint connects a parent body to a child body. A frame is fixed on each
// side: the parent-side joint frame (placed by X_PJ in the parent body frame)
// and the child-side joint frame (placed by X_CJ in the child body frame).
// A joint model describes the motion of the child-side frame relative to the
// parent-side one:
//
//   X_J(q)    placement of the child-side joint frame in the parent-side one;
//   S(q)      motion subspace: the relative velocity of the child-side frame,
//             as a motion vector in the child-side frame, is S(q) * v;
//   c(q, v)   the apparent time derivative of S in the child-side frame,
//             times v. Zero when S is constant in that frame.
//
// Coordinates q and velocities v need not have the same size (nq >= nv): a
// joint with a unit quaternion has four coordinates and three velocities.
// They are related by q_dot = G(q) * v, computed by q_dot(). Velocities are
// always body-fixed: they are expressed in the child-side joint frame.

#include "mbd/core/math.hpp"

namespace mbd::kernel {

/// 6 x nv matrix with nv <= 6, stored without heap allocation.
using Mat6X = Eigen::Matrix<Real, 6, Eigen::Dynamic, 0, 6, 6>;

using ConstVecRef = Eigen::Ref<const VecX>;
using VecRef      = Eigen::Ref<VecX>;

/// What a joint model computes for one configuration and velocity.
struct JointData {
    Transform3 X_J;                       ///< Child-side joint frame in the parent-side one
    Mat6X      S{Mat6X::Zero(6, 0)};      ///< Motion subspace, in the child-side joint frame
    Vec6       c{Vec6::Zero()};           ///< dS/dt * v, in the child-side joint frame
};

/// Interface of all joint models. Joint models hold no state and never change
/// after construction, so one instance can be shared by several models and
/// threads.
class JointModel {
public:
    virtual ~JointModel() = default;

    /// Short name, for messages and test output.
    virtual const char* name() const = 0;

    /// Number of coordinates.
    virtual int nq() const = 0;

    /// Number of velocities (degrees of freedom).
    virtual int nv() const = 0;

    /// X_J, S and c at configuration q and velocity v, which are this joint's
    /// segments of the system vectors (sizes nq and nv).
    virtual void calc(ConstVecRef q, ConstVecRef v, JointData& d) const = 0;

    /// q_dot = G(q) * v. The default is q_dot = v, for joints with nq == nv.
    virtual void q_dot(ConstVecRef q, ConstVecRef v, VecRef qd) const;

    /// q_out = q (+) dv: the configuration reached from q by moving at the
    /// constant velocity dv for unit time. q_out may be q. The default is
    /// q + dv, exact for joints whose q_dot is v.
    virtual void integrate(ConstVecRef q, ConstVecRef dv, VecRef q_out) const;

    /// dv = q1 (-) q0, the inverse of integrate: q0 (+) dv = q1. Rotations
    /// take the smallest angle. The default is q1 - q0.
    virtual void difference(ConstVecRef q0, ConstVecRef q1, VecRef dv) const;

    /// Restore any invariant of the coordinates after they were changed by
    /// integration, such as a unit quaternion. The default does nothing.
    virtual void normalize(VecRef q) const;

    /// Reference configuration (the identity). The default is all zeros.
    virtual void neutral(VecRef q) const;

protected:
    JointModel() = default;
};

// ============================================================================
// Concrete joints. The joint axis is the Z axis of the joint frame.
// ============================================================================

/// Rotation about Z. q = (angle).
class RevoluteJointModel final : public JointModel {
public:
    const char* name() const override { return "revolute"; }
    int nq() const override { return 1; }
    int nv() const override { return 1; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
};

/// Translation along Z. q = (displacement).
class PrismaticJointModel final : public JointModel {
public:
    const char* name() const override { return "prismatic"; }
    int nq() const override { return 1; }
    int nv() const override { return 1; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
};

/// No relative motion.
class FixedJointModel final : public JointModel {
public:
    const char* name() const override { return "fixed"; }
    int nq() const override { return 0; }
    int nv() const override { return 0; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
};

/// Rotation about Z, then about the rotated X: q = (angle_z, angle_x).
class UniversalJointModel final : public JointModel {
public:
    const char* name() const override { return "universal"; }
    int nq() const override { return 2; }
    int nv() const override { return 2; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
};

/// Rotation about Z and translation along Z: q = (angle, displacement).
class CylindricalJointModel final : public JointModel {
public:
    const char* name() const override { return "cylindrical"; }
    int nq() const override { return 2; }
    int nv() const override { return 2; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
};

/// Motion in the XY plane: q = (x, y, angle about Z), with x and y in the
/// parent-side joint frame.
class PlanarJointModel final : public JointModel {
public:
    const char* name() const override { return "planar"; }
    int nq() const override { return 3; }
    int nv() const override { return 3; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
};

/// Any rotation. q = unit quaternion (x, y, z, w), Eigen's storage order.
/// v = angular velocity in the child-side joint frame.
class SphericalJointModel final : public JointModel {
public:
    const char* name() const override { return "spherical"; }
    int nq() const override { return 4; }
    int nv() const override { return 3; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
    void q_dot(ConstVecRef q, ConstVecRef v, VecRef qd) const override;
    void integrate(ConstVecRef q, ConstVecRef dv, VecRef q_out) const override;
    void difference(ConstVecRef q0, ConstVecRef q1, VecRef dv) const override;
    void normalize(VecRef q) const override;
    void neutral(VecRef q) const override;
};

/// Any rigid motion. q = (translation in the parent-side joint frame,
/// unit quaternion (x, y, z, w)). v = (angular velocity, linear velocity of
/// the child-side frame origin), both in the child-side joint frame.
class FreeJointModel final : public JointModel {
public:
    const char* name() const override { return "free"; }
    int nq() const override { return 7; }
    int nv() const override { return 6; }
    void calc(ConstVecRef q, ConstVecRef v, JointData& d) const override;
    void q_dot(ConstVecRef q, ConstVecRef v, VecRef qd) const override;
    void integrate(ConstVecRef q, ConstVecRef dv, VecRef q_out) const override;
    void difference(ConstVecRef q0, ConstVecRef q1, VecRef dv) const override;
    void normalize(VecRef q) const override;
    void neutral(VecRef q) const override;
};

} // namespace mbd::kernel
