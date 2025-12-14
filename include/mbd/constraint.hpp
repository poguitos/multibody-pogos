#pragma once

#include "mbd/core.hpp"
#include "mbd/math.hpp"
#include "mbd/system.hpp"

namespace mbd {

// Abstract base class for holonomic constraints: Phi(q) = 0.
// This interface allows the solver to treat different joints (distance, revolute, etc.) uniformly.
class Constraint {
public:
    virtual ~Constraint() = default;

    // Return the number of scalar equations in this constraint (rows in Jacobian).
    // For a distance constraint, this is 1. For a spherical joint, it would be 3.
    virtual int equation_count() const = 0;

    // Compute the position error vector Phi(q).
    // Result 'phi' must be resized to equation_count().
    // A value of 0 means the constraint is satisfied.
    virtual void evaluate(const MultibodySystem& system, 
                          Eigen::VectorXd& phi) const = 0;

    // Compute Jacobian blocks J1 and J2.
    // The full Jacobian J relates velocity to the time derivative of Phi:
    // dot(Phi) = J * v = J1 * v1 + J2 * v2.
    // Dimensions of blocks: equation_count() x 6.
    virtual void jacobian(const MultibodySystem& system, 
                          Eigen::MatrixXd& J1, 
                          Eigen::MatrixXd& J2) const = 0;

    // Compute the velocity bias term gamma = dot(J) * v.
    // This represents the "acceleration" of the constraint error if joint accelerations were zero.
    // It allows us to solve the acceleration equation: J * a = -gamma - J * dot(v)_unconstrained
    // (Also known as the right-hand-side term for the acceleration constraint).
    virtual void velocity_bias(const MultibodySystem& system, 
                               Eigen::VectorXd& gamma) const = 0;

    // Indices of the bodies involved in this constraint.
    BodyIndex body1_idx;
    BodyIndex body2_idx;

protected:
    Constraint(BodyIndex b1, BodyIndex b2) : body1_idx(b1), body2_idx(b2) {}
};

// Constraint that maintains a fixed distance between two points on two bodies.
// Mathematically: Phi = || p2_W - p1_W || - length = 0
class DistanceConstraint : public Constraint {
public:
    Vec3 anchor1_B; // Attachment point on Body 1 (in local Body 1 frame)
    Vec3 anchor2_B; // Attachment point on Body 2 (in local Body 2 frame)
    Real target_distance; // The fixed length L0

    DistanceConstraint(BodyIndex b1, BodyIndex b2,
                       const Vec3& a1_local, const Vec3& a2_local,
                       Real dist)
        : Constraint(b1, b2)
        , anchor1_B(a1_local)
        , anchor2_B(a2_local)
        , target_distance(dist)
    {
        MBD_THROW_IF(target_distance < Real(0.0), "DistanceConstraint must have positive length");
    }

    int equation_count() const override { return 1; }

    // Evaluate position error: current_dist - target_dist
    void evaluate(const MultibodySystem& system, Eigen::VectorXd& phi) const override {
        // 1. Get states
        const RigidBodyState& s1 = system.states[body1_idx];
        const RigidBodyState& s2 = system.states[body2_idx];

        // 2. Compute world points
        Vec3 r1_W = s1.pose_WB().rotate(anchor1_B);
        Vec3 r2_W = s2.pose_WB().rotate(anchor2_B);
        
        Vec3 p1_W = s1.p_WB + r1_W;
        Vec3 p2_W = s2.p_WB + r2_W;

        // 3. Distance
        Real current_dist = (p2_W - p1_W).norm();
        phi.resize(1);
        phi(0) = current_dist - target_distance;
    }

    // Compute Jacobians J1 and J2
    void jacobian(const MultibodySystem& system, 
                  Eigen::MatrixXd& J1, 
                  Eigen::MatrixXd& J2) const override 
    {
        // Dimensions: 1 x 6
        J1.resize(1, 6);
        J2.resize(1, 6);

        const RigidBodyState& s1 = system.states[body1_idx];
        const RigidBodyState& s2 = system.states[body2_idx];

        // Transforms
        Vec3 r1_W = s1.pose_WB().rotate(anchor1_B);
        Vec3 r2_W = s2.pose_WB().rotate(anchor2_B);
        Vec3 p1_W = s1.p_WB + r1_W;
        Vec3 p2_W = s2.p_WB + r2_W;

        Vec3 diff = p2_W - p1_W;
        Real dist = diff.norm();

        // Normal vector n pointing from 1 to 2
        Vec3 n = Vec3::Zero();
        if (dist > Real(1e-12)) {
            n = diff / dist;
        }

        // Jacobian derivation:
        // dot(Phi) = n^T * (v_point2 - v_point1)
        // v_point = v_com + w x r
        
        // J1: [-n^T, -(r1 x n)^T] = [-n^T, (n x r1)^T]
        J1.block<1, 3>(0, 0) = -n.transpose();
        J1.block<1, 3>(0, 3) = -(r1_W.cross(n)).transpose();

        // J2: [ n^T,  (r2 x n)^T] = [ n^T, -(n x r2)^T]
        J2.block<1, 3>(0, 0) = n.transpose();
        J2.block<1, 3>(0, 3) = (r2_W.cross(n)).transpose();
    }

    // Compute velocity bias: gamma = dot(J) * v
    // For distance constraint, this represents the centripetal acceleration term.
    void velocity_bias(const MultibodySystem& system, Eigen::VectorXd& gamma) const override {
        const RigidBodyState& s1 = system.states[body1_idx];
        const RigidBodyState& s2 = system.states[body2_idx];
        
        // Geometry
        Vec3 r1 = s1.pose_WB().rotate(anchor1_B);
        Vec3 r2 = s2.pose_WB().rotate(anchor2_B);
        Vec3 p1 = s1.p_WB + r1;
        Vec3 p2 = s2.p_WB + r2;
        Vec3 diff = p2 - p1;
        Real dist = diff.norm();
        
        gamma.resize(1);
        if (dist < 1e-12) {
            gamma.setZero();
            return;
        }

        // Velocities of the attachment points
        // v_point = v_com + w x r
        Vec3 v1_point = s1.v_WB + s1.w_WB.cross(r1);
        Vec3 v2_point = s2.v_WB + s2.w_WB.cross(r2);
        
        // Relative velocity
        Vec3 v_rel = v2_point - v1_point;
        
        Vec3 n = diff / dist;
        
        // Scalar components of velocity parallel and perpendicular to the link
        Real v_dot_n = v_rel.dot(n);
        Real v_sq = v_rel.squaredNorm();
        
        // Standard formula for distance constraint bias:
        // gamma = ( |v_rel|^2 - (v_rel . n)^2 ) / distance
        // This effectively removes the centripetal component from the acceleration requirement.
        gamma(0) = (v_sq - v_dot_n * v_dot_n) / dist;
    }
};

} // namespace mbd