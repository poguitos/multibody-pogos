#pragma once

// Joint abstraction: coordinate generators for tree-based multibody dynamics.
//
// Each joint connects a parent body to a child body. It parameterizes
// the relative motion between them using generalized coordinates q.
//
// Convention:
//   - The joint axis is always the local Z-axis of the joint frame.
//   - X_PJ: transform from joint frame to parent body frame.
//   - X_CJ: transform from joint frame to child body frame.
//   - X_J(q): the motion across the joint, from child joint frame
//             to parent joint frame, parameterized by q.
//   - Full parent-to-child: X_PC(q) = X_PJ * X_J(q) * X_CJ.inverse()
//
//   - Motion subspace S (6 x n_dof):
//       V_rel = S * q_dot
//     where V_rel = [omega; v] in joint frame coordinates.

#include "mbd/core.hpp"
#include "mbd/math.hpp"

#include <Eigen/Dense>

namespace mbd {

/// Abstract base class for joints as coordinate generators.
class Joint {
public:
    virtual ~Joint() = default;

    /// Number of generalized coordinates (DOFs) for this joint.
    virtual int num_dof() const = 0;

    /// Relative transform across the joint given coordinates q.
    /// Returns X_J(q): from child joint frame to parent joint frame.
    /// q must have size num_dof().
    virtual Transform3 joint_transform(const VecX& q) const = 0;

    /// Motion subspace matrix (6 x num_dof) in the joint frame.
    /// Maps q_dot to 6D relative velocity [omega; v].
    /// For joints with constant S (revolute, prismatic, fixed), q is unused.
    virtual Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& q) const = 0;

    /// Time derivative of S * q_dot due to joint-frame kinematics.
    /// For most simple joints this is zero. Override for joints where
    /// S depends on q (e.g., spherical with Euler angles).
    virtual Vec6 bias_acceleration(const VecX& q, const VecX& q_dot) const
    {
        (void)q;
        (void)q_dot;
        return Vec6::Zero();
    }

    /// Compute the full parent-to-child transform:
    ///   X_PC(q) = X_PJ * X_J(q) * X_CJ.inverse()
    Transform3 parent_to_child_transform(const VecX& q) const
    {
        return X_PJ * joint_transform(q) * X_CJ.inverse();
    }

    // --- Data ---------------------------------------------------------------

    /// Transform from joint frame to parent body frame.
    Transform3 X_PJ;

    /// Transform from joint frame to child body frame.
    Transform3 X_CJ;

    /// Index of the parent body in the MultibodySystem.
    BodyIndex parent_body_idx{kGroundIndex};

    /// Index of the child body in the MultibodySystem.
    BodyIndex child_body_idx{kNoParent};

protected:
    Joint() = default;

    Joint(const Transform3& x_pj, const Transform3& x_cj,
          BodyIndex parent, BodyIndex child)
        : X_PJ(x_pj), X_CJ(x_cj)
        , parent_body_idx(parent), child_body_idx(child)
    {}
};

// ============================================================================
// Revolute joint: 1 DOF rotation about the joint Z-axis
// ============================================================================

class RevoluteCoordJoint : public Joint {
public:
    /// Construct a revolute joint.
    /// \param x_pj  Joint frame expressed in parent body frame.
    /// \param x_cj  Joint frame expressed in child body frame.
    /// \param parent Index of parent body.
    /// \param child  Index of child body.
    RevoluteCoordJoint(const Transform3& x_pj, const Transform3& x_cj,
                       BodyIndex parent, BodyIndex child)
        : Joint(x_pj, x_cj, parent, child)
    {}

    int num_dof() const override { return 1; }

    Transform3 joint_transform(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 1);
        const Real theta = q(0);
        Quat q_rot(Eigen::AngleAxisd(theta, Vec3::UnitZ()));
        return Transform3(q_rot, Vec3::Zero());
    }

    Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& /*q*/) const override
    {
        Eigen::Matrix<Real, 6, 1> S;
        S << 0.0, 0.0, 1.0,   // omega about Z
             0.0, 0.0, 0.0;   // no linear velocity
        return S;
    }
};

// ============================================================================
// Prismatic joint: 1 DOF translation along the joint Z-axis
// ============================================================================

class PrismaticCoordJoint : public Joint {
public:
    PrismaticCoordJoint(const Transform3& x_pj, const Transform3& x_cj,
                        BodyIndex parent, BodyIndex child)
        : Joint(x_pj, x_cj, parent, child)
    {}

    int num_dof() const override { return 1; }

    Transform3 joint_transform(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 1);
        const Real d = q(0);
        return Transform3(Quat::Identity(), Vec3(0.0, 0.0, d));
    }

    Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& /*q*/) const override
    {
        Eigen::Matrix<Real, 6, 1> S;
        S << 0.0, 0.0, 0.0,   // no angular velocity
             0.0, 0.0, 1.0;   // linear along Z
        return S;
    }
};

// ============================================================================
// Fixed joint: 0 DOF — rigidly connects parent to child
// ============================================================================

class FixedJoint : public Joint {
public:
    FixedJoint(const Transform3& x_pj, const Transform3& x_cj,
               BodyIndex parent, BodyIndex child)
        : Joint(x_pj, x_cj, parent, child)
    {}

    int num_dof() const override { return 0; }

    Transform3 joint_transform(const VecX& /*q*/) const override
    {
        return Transform3::Identity();
    }

    Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& /*q*/) const override
    {
        // 6 x 0 matrix
        return Eigen::Matrix<Real, 6, Eigen::Dynamic>(6, 0);
    }
};

} // namespace mbd