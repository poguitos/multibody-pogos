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

} // namespace mbd
