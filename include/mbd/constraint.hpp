#pragma once

#include "mbd/core.hpp"
#include "mbd/math.hpp"
#include "mbd/system.hpp"

namespace mbd {

// Abstract base class for holonomic constraints: Phi(q) = 0.
class Constraint {
public:
    virtual ~Constraint() = default;

    // Return the number of scalar equations in this constraint (rows in Jacobian).
    virtual int equation_count() const = 0;

    // Compute the position error vector Phi(q).
    // result size must be equation_count().
    virtual void evaluate(const MultibodySystem& system, 
                          Eigen::VectorXd& phi) const = 0;

    // Compute Jacobian blocks J1 and J2 (6 columns each).
    // J1 corresponds to body_index_1, J2 to body_index_2.
    // Dimensions of blocks: equation_count() x 6.
    // If a body is ground (index -1 or similar), the block is zero/ignored.
    virtual void jacobian(const MultibodySystem& system, 
                          Eigen::MatrixXd& J1, 
                          Eigen::MatrixXd& J2) const = 0;

    // Indices of the bodies involved.
    BodyIndex body1_idx;
    BodyIndex body2_idx;

protected:
    Constraint(BodyIndex b1, BodyIndex b2) : body1_idx(b1), body2_idx(b2) {}
};

// Constraint that maintains a fixed distance between two points on two bodies.
// Phi = || p2_W - p1_W || - length = 0
class DistanceConstraint : public Constraint {
public:
    Vec3 anchor1_B; // Attachment on Body 1 (local)
    Vec3 anchor2_B; // Attachment on Body 2 (local)
    Real target_distance;

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
        // dot(Phi) = n^T * (v2 + w2 x r2 - v1 - w1 x r1)
        //          = n^T*v2 + n^T*(w2 x r2) - n^T*v1 - n^T*(w1 x r1)
        // Using vector triple product identity a . (b x c) = (a x b) . c = (c x a) . b
        // n . (w x r) = w . (r x n)
        // So coeff of w is (r x n)^T
        
        // J1: [-n^T, -(r1 x n)^T] = [-n^T, (n x r1)^T]
        J1.block<1, 3>(0, 0) = -n.transpose();
        J1.block<1, 3>(0, 3) = -(r1_W.cross(n)).transpose();

        // J2: [ n^T,  (r2 x n)^T] = [ n^T, -(n x r2)^T]
        J2.block<1, 3>(0, 0) = n.transpose();
        J2.block<1, 3>(0, 3) = (r2_W.cross(n)).transpose();
    }
};

} // namespace mbd