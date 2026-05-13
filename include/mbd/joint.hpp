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

    /// Starting index of this joint's coordinates in the system q vector.
    /// Set by MultibodySystem::add_joint().
    int q_offset{-1};

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

// ============================================================================
// Exponential map helpers (used by SphericalCoordJoint and FreeCoordJoint)
// ============================================================================

namespace detail {

/// Rodrigues' formula: rotation matrix from rotation vector r.
inline Mat3 exp_map_rotation(const Vec3& r)
{
    const Real theta = r.norm();
    if (theta < Real(1e-10)) {
        return Mat3::Identity() + skew(r);
    }
    const Vec3 axis = r / theta;
    return Eigen::AngleAxisd(theta, axis).toRotationMatrix();
}

/// Right Jacobian of SO(3): maps r_dot to angular velocity in the
/// ROTATED (body/child) frame.
///   omega_body = J_R(r) * r_dot
///
/// This is the correct Jacobian for our convention because the FK code
/// rotates the result by q_WJ which includes the joint rotation R_J(q).
/// Composing: R_J * omega_body = R_J * J_R * r_dot = J_L * r_dot = omega_parent_frame.
///
/// J_R(r) = I - (1 - cos(theta))/theta^2 * [r]x + (theta - sin(theta))/theta^3 * [r]x^2
inline Mat3 exp_map_jacobian(const Vec3& r)
{
    const Real theta = r.norm();
    if (theta < Real(1e-10)) {
        // Taylor: J_R = I - 0.5 * [r]x + ...
        return Mat3::Identity() - Real(0.5) * skew(r);
    }
    const Real th2 = theta * theta;
    const Real th3 = th2 * theta;
    const Mat3 rx  = skew(r);
    return Mat3::Identity()
         - ((Real(1.0) - std::cos(theta)) / th2) * rx
         + ((theta - std::sin(theta)) / th3) * (rx * rx);
}

/// Time derivative of E(r) * r_dot, evaluated at (r, r_dot).
/// Returns the bias acceleration contribution for the spherical joint:
///   c = dE/dt * r_dot = (dE/dr * r_dot) * r_dot
/// This is needed for RNEA when S depends on q.
///
/// Computed by finite difference internally for robustness.
inline Vec3 exp_map_bias(const Vec3& r, const Vec3& r_dot)
{
    const Real eps = Real(1e-7);
    Vec3 bias = Vec3::Zero();
    for (int k = 0; k < 3; ++k) {
        Vec3 r_plus = r;
        r_plus(k) += eps;
        Vec3 r_minus = r;
        r_minus(k) -= eps;
        const Mat3 E_plus  = exp_map_jacobian(r_plus);
        const Mat3 E_minus = exp_map_jacobian(r_minus);
        const Mat3 dE_drk  = (E_plus - E_minus) / (Real(2.0) * eps);
        bias += dE_drk * r_dot * r_dot(k);
    }
    return bias;
}

} // namespace detail

// ============================================================================
// Spherical joint: 3 DOF rotation, parameterized by rotation vector
// ============================================================================
//
// q = [rx, ry, rz]: rotation vector (exponential map).
// At q = 0 the child frame is aligned with the joint frame.

class SphericalCoordJoint : public Joint {
public:
    SphericalCoordJoint(const Transform3& x_pj, const Transform3& x_cj,
                        BodyIndex parent, BodyIndex child)
        : Joint(x_pj, x_cj, parent, child)
    {}

    int num_dof() const override { return 3; }

    Transform3 joint_transform(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 3);
        const Vec3 r(q(0), q(1), q(2));
        const Mat3 R = detail::exp_map_rotation(r);
        return Transform3(R, Vec3::Zero());
    }

    Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 3);
        const Vec3 r(q(0), q(1), q(2));
        const Mat3 E = detail::exp_map_jacobian(r);

        Eigen::Matrix<Real, 6, 3> S;
        S.setZero();
        S.topRows(3) = E;  // angular part: omega = E * r_dot
        return S;
    }

    Vec6 bias_acceleration(const VecX& q, const VecX& q_dot) const override
    {
        MBD_ASSERT(q.size() == 3 && q_dot.size() == 3);
        const Vec3 r(q(0), q(1), q(2));
        const Vec3 r_dot(q_dot(0), q_dot(1), q_dot(2));

        Vec6 bias = Vec6::Zero();
        bias.head<3>() = detail::exp_map_bias(r, r_dot);
        return bias;
    }
};

// ============================================================================
// Universal joint: 2 DOF rotation about Z then rotated X
// ============================================================================
//
// q = [theta_z, theta_x]: first rotate about joint Z by q(0),
// then about the (rotated) X-axis by q(1).

class UniversalCoordJoint : public Joint {
public:
    UniversalCoordJoint(const Transform3& x_pj, const Transform3& x_cj,
                        BodyIndex parent, BodyIndex child)
        : Joint(x_pj, x_cj, parent, child)
    {}

    int num_dof() const override { return 2; }

    Transform3 joint_transform(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 2);
        const Quat q_z(Eigen::AngleAxisd(q(0), Vec3::UnitZ()));
        const Quat q_x(Eigen::AngleAxisd(q(1), Vec3::UnitX()));
        return Transform3(q_z * q_x, Vec3::Zero());
    }

    Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 2);

        // omega = [0,0,1] * q_dot(0) + Rz(q0) * [1,0,0] * q_dot(1)
        // S = [ [0,0,1]^T | Rz(q0)*[1,0,0]^T ]  (angular rows)
        // Linear rows are zero (pure rotation).
        const Real c0 = std::cos(q(0));
        const Real s0 = std::sin(q(0));

        Eigen::Matrix<Real, 6, 2> S;
        S.setZero();

        // Column 0: rotation about Z
        S(2, 0) = Real(1.0);

        // Column 1: rotation about the rotated X-axis = Rz(q0) * [1,0,0]
        S(0, 1) = c0;
        S(1, 1) = s0;

        return S;
    }

    Vec6 bias_acceleration(const VecX& q, const VecX& q_dot) const override
    {
        MBD_ASSERT(q.size() == 2 && q_dot.size() == 2);

        // S depends on q(0), so dS/dt * q_dot is non-zero.
        // d/dt(Rz(q0)*[1,0,0]) = q_dot(0) * [-sin(q0), cos(q0), 0]
        // bias = dS/dt * q_dot, only column 1 contributes:
        //   bias_angular = q_dot(0) * [-sin(q0), cos(q0), 0] * q_dot(1)
        const Real s0 = std::sin(q(0));
        const Real c0 = std::cos(q(0));

        Vec6 bias = Vec6::Zero();
        bias(0) = -s0 * q_dot(0) * q_dot(1);
        bias(1) =  c0 * q_dot(0) * q_dot(1);
        return bias;
    }
};

// ============================================================================
// Free joint: 6 DOF (3 translation + 3 rotation)
// ============================================================================
//
// q = [tx, ty, tz, rx, ry, rz]:
//   First three: translation in the joint frame.
//   Last three: rotation vector (exponential map).
//
// This joint is used for a floating body (e.g., vehicle chassis).

class FreeCoordJoint : public Joint {
public:
    FreeCoordJoint(const Transform3& x_pj, const Transform3& x_cj,
                   BodyIndex parent, BodyIndex child)
        : Joint(x_pj, x_cj, parent, child)
    {}

    int num_dof() const override { return 6; }

    Transform3 joint_transform(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 6);
        const Vec3 t(q(0), q(1), q(2));
        const Vec3 r(q(3), q(4), q(5));
        const Mat3 R = detail::exp_map_rotation(r);
        return Transform3(R, t);
    }

    Eigen::Matrix<Real, 6, Eigen::Dynamic>
    motion_subspace(const VecX& q) const override
    {
        MBD_ASSERT(q.size() == 6);
        const Vec3 r(q(3), q(4), q(5));
        const Mat3 E   = detail::exp_map_jacobian(r);
        const Mat3 R_J = detail::exp_map_rotation(r);

        Eigen::Matrix<Real, 6, 6> S;
        S.setZero();

        // Columns 0-2: translation (linear velocity in PARENT frame).
        // Algorithms multiply by R_WJ = R_WP * R_PJ * R_J(q), so to get
        // v_rel_W = R_WP * R_PJ * q_dot(0:3) (parent-frame velocity in world),
        // we need S_lin = R_J^T * R_PJ^T. For our usage X_PJ = identity, so
        // R_PJ = I and S_lin = R_J^T.
        // This convention is consistent with q(0:3) being parent-frame
        // translation as set in joint_transform().
        S.block<3,3>(3, 0) = R_J.transpose();

        // Columns 3-5: rotation (angular velocity via exponential map Jacobian)
        S.block<3,3>(0, 3) = E;

        return S;
    }

    Vec6 bias_acceleration(const VecX& q, const VecX& q_dot) const override
    {
        MBD_ASSERT(q.size() == 6 && q_dot.size() == 6);
        const Vec3 r(q(3), q(4), q(5));
        const Vec3 r_dot(q_dot(3), q_dot(4), q_dot(5));
        const Vec3 t_dot(q_dot(0), q_dot(1), q_dot(2));

        Vec6 bias = Vec6::Zero();

        // Angular bias: dE/dt * r_dot (exponential map S depends on r)
        bias.head<3>() = detail::exp_map_bias(r, r_dot);

        // Linear bias: dS_lin/dt * q_dot(0:3) where S_lin = R_J^T.
        // Derivation: we want a_rel_lin_W (joint-frame contribution to body
        // acceleration in world) to equal q_ddot(0:3) when no external effects.
        // Algorithm computes a_rel_lin_W = R_WJ * (S_lin * q_ddot(0:3) + c_bias_lin)
        // and adds w_body × v_rel_W. Working through the math (see derivation
        // notes), the linear bias must be:
        //   c_bias_lin = -skew(E*r_dot) * R_J^T * t_dot
        // This cancels the spurious w × v term that would otherwise appear
        // when q_dot is constant in the parent frame.
        const Mat3 E   = detail::exp_map_jacobian(r);
        const Mat3 R_J = detail::exp_map_rotation(r);
        const Vec3 omega_J = E * r_dot;
        const Vec3 v_in_joint = R_J.transpose() * t_dot;

        bias.tail<3>() = -omega_J.cross(v_in_joint);

        return bias;
    }
};
} // namespace mbd