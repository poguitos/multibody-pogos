#pragma once

// Core math types and utilities for the multibody solver.
// This header is intended to be header-only.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>

namespace mbd {

//------------------------------------------------------------------------------
// Scalar and index types
//------------------------------------------------------------------------------

using Real  = double;          // Main scalar type for dynamics
using Index = Eigen::Index;    // For sizes/indices compatible with Eigen

//------------------------------------------------------------------------------
// Fixed-size vector and matrix aliases
//------------------------------------------------------------------------------

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;

// Rotations: 3x3 rotation matrix and unit quaternion
using RotMat3 = Eigen::Matrix3d;
using Quat    = Eigen::Quaterniond;

// 6D spatial types (we'll use them later for spatial algebra)
using Vec6 = Eigen::Matrix<Real, 6, 1>;
using Mat6 = Eigen::Matrix<Real, 6, 6>;

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

// Gravitational acceleration magnitude (SI units: m/s^2).
// We keep it simple for now; later we can make it configurable per scenario.
inline constexpr Real g = 9.81;

//------------------------------------------------------------------------------
// Basic helpers
//------------------------------------------------------------------------------

// Build the 3x3 skew-symmetric matrix associated with a 3D vector v:
//   skew(v) * w = v x w
inline Mat3 skew(const Vec3& v)
{
    Mat3 S;
    S <<   Real(0),  -v.z(),   v.y(),
           v.z(),    Real(0), -v.x(),
          -v.y(),    v.x(),   Real(0);
    return S;
}

// Safely normalize a quaternion; if the norm is too small, return identity.
inline Quat normalize_quat(const Quat& q)
{
    const Real n = q.norm();
    if (n < Real(1e-12)) {
        return Quat::Identity();
    }
    return Quat(q.coeffs() / n);
}

// Convert a 3D angular velocity omega (in body frame) and time step dt
// into a small rotation quaternion using the exponential map.
inline Quat delta_rotation_from_omega(const Vec3& omega, Real dt)
{
    const Vec3 theta = omega * dt;
    const Real angle = theta.norm();

    if (angle < Real(1e-12)) {
        // Very small rotation: use first-order approximation
        return Quat::Identity();
    }

    const Vec3 axis = theta / angle;
    const Real half = Real(0.5) * angle;
    const Real s = std::sin(half);

    return Quat(std::cos(half),
                axis.x() * s,
                axis.y() * s,
                axis.z() * s);
}

// Apply small rotation represented by angular velocity*dt to a quaternion.
// This is a building block for simple orientation integration schemes.
inline Quat integrate_quat(const Quat& q, const Vec3& omega, Real dt)
{
    Quat dq = delta_rotation_from_omega(omega, dt);
    Quat result = dq * q;
    return normalize_quat(result);
}
// Rigid transform in 3D: rotation + translation.
// Convention: this transform maps coordinates from a local frame B to world W.
// For a point x_B expressed in frame B, the world coordinates are:
//     x_W = R * x_B + p
struct Transform3 {
    Mat3 R;  // rotation from local to world
    Vec3 p;  // origin of local frame expressed in world coordinates

    // Default: identity transform
    Transform3()
        : R(Mat3::Identity()), p(Vec3::Zero())
    {}

    Transform3(const Mat3& R_in, const Vec3& p_in)
        : R(R_in), p(p_in)
    {}

    static Transform3 Identity() {
        return Transform3(Mat3::Identity(), Vec3::Zero());
    }

    // Apply transform to a point in the local frame.
    // x_W = R * x_L + p
    Vec3 apply(const Vec3& x_local) const {
        return R * x_local + p;
    }

    // Inverse transform: from world back to local.
    // For x_W = R * x_L + p, we have x_L = R^T * (x_W - p).
    Transform3 inverse() const {
        Mat3 R_inv = R.transpose();
        Vec3 p_inv = -R_inv * p;
        return Transform3(R_inv, p_inv);
    }
};

// Composition of transforms.
// If T1 maps from A to B, and T2 maps from B to C,
// then (T1 * T2) maps from A to C.
//
// For points: (T1 * T2) * x == T1 * (T2 * x)
inline Transform3 operator*(const Transform3& T1, const Transform3& T2) {
    Transform3 result;
    result.R = T1.R * T2.R;
    result.p = T1.R * T2.p + T1.p;
    return result;
}

// Apply transform to a point using operator*.
// x_W = T * x_L
inline Vec3 operator*(const Transform3& T, const Vec3& x_local) {
    return T.apply(x_local);
}

//------------------------------------------------------------------------------
// Frame-to-frame helpers based on Transform3
//------------------------------------------------------------------------------

/// Compute the relative transform X_AB given world-frame transforms X_WA and X_WB.
/// By convention, X_WA maps a point from frame A to world:
///   p_W = X_WA.apply(p_A)
/// and X_AB maps from B to A:
///   p_A = X_AB.apply(p_B)
/// so:
///   X_AB = X_AW * X_WB = X_WA.inverse() * X_WB
inline Transform3 ComputeRelativeTransform(const Transform3& X_WA,
                                           const Transform3& X_WB)
{
    return X_WA.inverse() * X_WB;
}

/// Convenience wrapper: transform a point from frame B to frame A given X_AB.
/// This is simply a named wrapper around Transform3::apply.
inline Vec3 TransformPoint(const Transform3& X_AB, const Vec3& p_B)
{
    return X_AB.apply(p_B);
}

/// Transform a point expressed in frame B into frame A using the world-frame
/// poses of A and B.
/// Given:
///   p_W = X_WA.apply(p_A)
///   p_W = X_WB.apply(p_B)
/// we want p_A such that:
///   p_A = X_AB.apply(p_B)   with   X_AB = X_WA.inverse() * X_WB.
inline Vec3 TransformPointBetweenFrames(const Transform3& X_WA,
                                        const Transform3& X_WB,
                                        const Vec3& p_B)
{
    const Transform3 X_AB = ComputeRelativeTransform(X_WA, X_WB);
    return X_AB.apply(p_B);
}

} // namespace mbd
