#pragma once

// Core math types and utilities for the multibody solver.
// This header is intended to be header-only.
//
// ============================================================================
//  CONVENTIONS (L0.1.2)
// ============================================================================
//
//  Quaternion convention (Hamilton, Eigen default):
//    - Storage order in Eigen: (x, y, z, w) internally, but the constructor
//      is Quat(w, x, y, z).
//    - q_WB rotates vectors FROM frame B TO frame W:
//        v_W = q_WB * v_B * q_WB_conj
//      or equivalently:
//        v_W = q_WB.toRotationMatrix() * v_B
//    - Composition: q_WC = q_WB * q_BC  (right-to-left, same as matrices).
//
//  Transform3 convention:
//    - T_WB maps points from local frame B to world frame W:
//        x_W = T_WB.apply(x_B) = T_WB.q * x_B + T_WB.p
//    - T_WB.q is the rotation from B to W.
//    - T_WB.p is the origin of frame B expressed in W.
//    - Composition: T_WC = T_WB * T_BC.
//    - Inverse: T_BW = T_WB.inverse().
//
// ============================================================================

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>

namespace mbd
{
//------------------------------------------------------------------------------
// Scalar and index types
//------------------------------------------------------------------------------

using Real  = double;
using Index = Eigen::Index;

//------------------------------------------------------------------------------
// Fixed-size vector and matrix aliases
//------------------------------------------------------------------------------

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;

using RotMat3 = Eigen::Matrix3d;
using Quat    = Eigen::Quaterniond;

// 6D spatial types (used later for spatial algebra)
using Vec6 = Eigen::Matrix<Real, 6, 1>;
using Mat6 = Eigen::Matrix<Real, 6, 6>;

// Dynamic-size types (for generalized coordinates, Jacobians, etc.)
using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

inline constexpr Real g_accel = 9.81;

//------------------------------------------------------------------------------
// Basic helpers
//------------------------------------------------------------------------------

/// Build the 3x3 skew-symmetric matrix so that skew(v) * w == v.cross(w).
inline Mat3 skew(const Vec3& v)
{
    Mat3 S;
    S << Real(0),  -v.z(),   v.y(),
         v.z(),    Real(0), -v.x(),
        -v.y(),    v.x(),   Real(0);
    return S;
}

/// Safely normalize a quaternion; returns identity if the norm is degenerate.
inline Quat normalize_quat(const Quat& q)
{
    const Real n = q.norm();
    if (n < Real(1e-12)) {
        return Quat::Identity();
    }
    return Quat(q.w() / n, q.x() / n, q.y() / n, q.z() / n);
}

/// Exponential map: angular velocity * dt -> small rotation quaternion.
inline Quat delta_rotation_from_omega(const Vec3& omega, Real dt)
{
    const Vec3 theta = omega * dt;
    const Real angle = theta.norm();
    const Real half  = angle * Real(0.5);

    Real s_over_angle;

    if (angle < Real(1e-12)) {
        const Real half_sq = half * half;
        s_over_angle = Real(0.5) * (Real(1.0) - half_sq * Real(1.0 / 6.0));
    } else {
        s_over_angle = std::sin(half) / angle;
    }

    return Quat(std::cos(half),
                theta.x() * s_over_angle,
                theta.y() * s_over_angle,
                theta.z() * s_over_angle);
}

/// Apply a small rotation (omega * dt) to a quaternion and re-normalize.
inline Quat integrate_quat(const Quat& q, const Vec3& omega, Real dt)
{
    Quat dq = delta_rotation_from_omega(omega, dt);
    Quat result = dq * q;
    return normalize_quat(result);
}

//------------------------------------------------------------------------------
// Transform3: rigid transform in 3D  (Quat + Vec3 storage)
//------------------------------------------------------------------------------
//
// Convention: maps from local frame B to world W.
//   x_W = q * x_B + p
//
struct Transform3
{
    Quat q; // rotation from local to world
    Vec3 p; // origin of local frame expressed in world coordinates

    /// Default: identity transform.
    Transform3()
        : q(Quat::Identity())
        , p(Vec3::Zero())
    {}

    /// Construct from quaternion and translation.
    /// The quaternion is normalized to protect the rotation invariant.
    Transform3(const Quat& q_in, const Vec3& p_in)
        : q(normalize_quat(q_in))
        , p(p_in)
    {}

    /// Construct from rotation matrix and translation.
    /// Converts the matrix to a quaternion internally.
    Transform3(const Mat3& R_in, const Vec3& p_in)
        : q(normalize_quat(Quat(R_in)))
        , p(p_in)
    {}

    /// Identity transform (named constructor).
    static Transform3 Identity()
    {
        return Transform3{};
    }

    // --- Named factories ----------------------------------------------------

    static Transform3 FromRotationTranslation(const Mat3& R, const Vec3& t)
    {
        return Transform3(R, t);
    }

    static Transform3 FromQuatTranslation(const Quat& q_in, const Vec3& t)
    {
        return Transform3(q_in, t);
    }

    static Transform3 FromTranslation(const Vec3& t)
    {
        return Transform3(Quat::Identity(), t);
    }

    static Transform3 FromRotation(const Mat3& R)
    {
        return Transform3(R, Vec3::Zero());
    }

    static Transform3 FromRotation(const Quat& q_in)
    {
        return Transform3(q_in, Vec3::Zero());
    }

    // --- Accessors ----------------------------------------------------------

    /// Rotation as a quaternion (the internal representation).
    Quat&       rotation()       { return q; }
    const Quat& rotation() const { return q; }

    /// Translation.
    Vec3&       translation()       { return p; }
    const Vec3& translation() const { return p; }

    /// Rotation as a 3x3 matrix (derived -- computed on each call).
    Mat3 rotation_matrix() const { return q.toRotationMatrix(); }

    // --- Point / vector operations ------------------------------------------

    /// Transform a point from local to world: x_W = q * x_B + p.
    Vec3 apply(const Vec3& x_local) const
    {
        return q * x_local + p;
    }

    /// Rotate a direction vector (ignoring translation): v_W = q * v_B.
    Vec3 rotate(const Vec3& v_local) const
    {
        return q * v_local;
    }

    /// Inverse transform: from world back to local.
    Transform3 inverse() const
    {
        Quat q_inv = q.conjugate();
        Vec3 p_inv = -(q_inv * p);
        return Transform3(q_inv, p_inv);
    }
};

// --- Free operators ---------------------------------------------------------

/// Composition: if T1 maps A->B and T2 maps B->C, then T1*T2 maps A->C.
inline Transform3 operator*(const Transform3& T1, const Transform3& T2)
{
    return Transform3(T1.q * T2.q,
                      T1.q * T2.p + T1.p);
}

/// Apply transform to a point: x_W = T * x_B.
inline Vec3 operator*(const Transform3& T, const Vec3& x_local)
{
    return T.apply(x_local);
}

//------------------------------------------------------------------------------
// Frame-to-frame helpers
//------------------------------------------------------------------------------

inline Transform3 ComputeRelativeTransform(const Transform3& X_WA,
                                           const Transform3& X_WB)
{
    return X_WA.inverse() * X_WB;
}

inline Vec3 TransformPoint(const Transform3& X_AB, const Vec3& p_B)
{
    return X_AB.apply(p_B);
}

inline Vec3 TransformPointBetweenFrames(const Transform3& X_WA,
                                        const Transform3& X_WB,
                                        const Vec3&       p_B)
{
    return ComputeRelativeTransform(X_WA, X_WB).apply(p_B);
}

} // namespace mbd