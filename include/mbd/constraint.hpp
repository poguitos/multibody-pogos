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

// A Revolute Joint (Hinge) constrains two bodies to share a common point (anchor)
// and a common axis of rotation. Removes 5 DOFs.
class RevoluteJoint : public Constraint {
public:
    Vec3 anchor1_B; // Anchor on Body 1
    Vec3 axis1_B;   // Axis of rotation on Body 1 (must be normalized)
    
    Vec3 anchor2_B; // Anchor on Body 2
    Vec3 axis2_B;   // Axis of rotation on Body 2 (must be normalized)

    // Precomputed orthogonal basis vectors for Body 2's axis.
    // axis2_B, u2_B, v2_B form an orthonormal basis.
    Vec3 u2_B;
    Vec3 v2_B;

    RevoluteJoint(BodyIndex b1, BodyIndex b2,
                  const Vec3& a1, const Vec3& axis1,
                  const Vec3& a2, const Vec3& axis2)
        : Constraint(b1, b2)
        , anchor1_B(a1)
        , axis1_B(axis1.normalized())
        , anchor2_B(a2)
        , axis2_B(axis2.normalized())
    {
        // Generate orthogonal basis u2, v2 for axis2
        // Find a vector not parallel to axis2
        Vec3 temp = Vec3::UnitX();
        if (std::abs(axis2_B.dot(temp)) > 0.9) {
            temp = Vec3::UnitY();
        }
        u2_B = axis2_B.cross(temp).normalized();
        v2_B = axis2_B.cross(u2_B).normalized();
    }

    int equation_count() const override { return 5; }

    void evaluate(const MultibodySystem& system, Eigen::VectorXd& phi) const override {
        const RigidBodyState& s1 = system.states[body1_idx];
        const RigidBodyState& s2 = system.states[body2_idx];

        phi.resize(5);

        // --- 1. Position Constraints (3 eqs) ---
        // p2_W - p1_W = 0
        Vec3 r1_W = s1.pose_WB().rotate(anchor1_B);
        Vec3 r2_W = s2.pose_WB().rotate(anchor2_B);
        Vec3 p1_W = s1.p_WB + r1_W;
        Vec3 p2_W = s2.p_WB + r2_W;
        
        Vec3 pos_error = p2_W - p1_W;
        phi.segment<3>(0) = pos_error;

        // --- 2. Orientation Constraints (2 eqs) ---
        // axis1_W must be perpendicular to u2_W and v2_W
        Vec3 axis1_W = s1.pose_WB().rotate(axis1_B);
        Vec3 u2_W    = s2.pose_WB().rotate(u2_B);
        Vec3 v2_W    = s2.pose_WB().rotate(v2_B);

        phi(3) = axis1_W.dot(u2_W);
        phi(4) = axis1_W.dot(v2_W);
    }

    void jacobian(const MultibodySystem& system, 
                  Eigen::MatrixXd& J1, 
                  Eigen::MatrixXd& J2) const override 
    {
        J1.resize(5, 6);
        J2.resize(5, 6);
        J1.setZero();
        J2.setZero();

        const RigidBodyState& s1 = system.states[body1_idx];
        const RigidBodyState& s2 = system.states[body2_idx];

        // Transforms
        Vec3 r1_W = s1.pose_WB().rotate(anchor1_B);
        Vec3 r2_W = s2.pose_WB().rotate(anchor2_B);

        // --- Position Rows (0,1,2) ---
        // Similar to spherical joint
        // J1_pos = [-I,  skew(r1)] (Note: skew(r1)*w = w x r1 = -r1 x w. Formula is v + w x r. 
        // d/dt(p + r) = v + w x r = v - r x w.
        // So J_w term is -skew(r).
        // Let's re-verify sign.
        // Phi = p2 + r2 - p1 - r1
        // dot(Phi) = v2 - r2 x w2 - v1 + r1 x w1
        // J1 (coeff of v1, w1): [-I,  skew(r1)]
        // J2 (coeff of v2, w2): [ I, -skew(r2)]
        
        J1.block<3,3>(0,0) = -Mat3::Identity();
        J1.block<3,3>(0,3) = skew(r1_W);
        
        J2.block<3,3>(0,0) = Mat3::Identity();
        J2.block<3,3>(0,3) = -skew(r2_W);

        // --- Orientation Rows (3,4) ---
        Vec3 a1_W = s1.pose_WB().rotate(axis1_B);
        Vec3 u2_W = s2.pose_WB().rotate(u2_B);
        Vec3 v2_W = s2.pose_WB().rotate(v2_B);

        // Eq 3: Phi = a1 . u2 = 0
        // dot(Phi) = dot(a1) . u2 + a1 . dot(u2)
        //          = (w1 x a1) . u2 + a1 . (w2 x u2)
        //          = w1 . (a1 x u2) + w2 . (u2 x a1)
        //          = w1 . (a1 x u2) - w2 . (a1 x u2)
        // Let n1 = a1 x u2
        Vec3 n1 = a1_W.cross(u2_W);
        
        // J1 row 3 (angular part only, linear is 0) -> n1^T
        J1.block<1,3>(3,3) = n1.transpose();
        // J2 row 3 -> -n1^T
        J2.block<1,3>(3,3) = -n1.transpose();

        // Eq 4: Phi = a1 . v2 = 0
        // Let n2 = a1 x v2
        Vec3 n2 = a1_W.cross(v2_W);
        
        J1.block<1,3>(4,3) = n2.transpose();
        J2.block<1,3>(4,3) = -n2.transpose();
    }

    void velocity_bias(const MultibodySystem& system, Eigen::VectorXd& gamma) const override {
        gamma.resize(5);
        
        const RigidBodyState& s1 = system.states[body1_idx];
        const RigidBodyState& s2 = system.states[body2_idx];

        // --- Position Bias ---
        // gamma_pos = (w1 x (w1 x r1)) - (w2 x (w2 x r2))
        // Because J*a terms handle the linear a and alpha x r terms.
        // The bias is the "velocity dependent acceleration".
        // acc_point = a + alpha x r + w x (w x r)
        // Constraint: acc_p2 - acc_p1 = 0
        // (a2 + alpha2 x r2 + w2 x w2 x r2) - (a1 + ... ) = 0
        // J*a terms are (a2 - skew(r2)alpha2) - (a1 - skew(r1)alpha1)
        // Remaining terms moved to RHS (-gamma): -(w2 x w2 x r2 - w1 x w1 x r1)
        // So gamma = w2 x (w2 x r2) - w1 x (w1 x r1)
        
        Vec3 r1_W = s1.pose_WB().rotate(anchor1_B);
        Vec3 r2_W = s2.pose_WB().rotate(anchor2_B);
        
        Vec3 w1 = s1.w_WB;
        Vec3 w2 = s2.w_WB;

        Vec3 bias_pos = w2.cross(w2.cross(r2_W)) - w1.cross(w1.cross(r1_W));
        gamma.segment<3>(0) = bias_pos;

        // --- Orientation Bias ---
        // Eq: a1 . u2 = 0
        // 2nd deriv: d/dt ( w1.(a1xu2) - w2.(a1xu2) )
        // This is getting complex analytically.
        // Ideally: gamma = dot(J) * v
        // J row is [0, (a1 x u2)^T]. 
        // dot(J) row is [0, d/dt(a1 x u2)^T]
        // d/dt(a1 x u2) = dot(a1) x u2 + a1 x dot(u2)
        //               = (w1 x a1) x u2 + a1 x (w2 x u2)
        // gamma_i = (d/dt(n) . w1) - (d/dt(n) . w2) 
        //         = d/dt(n) . (w1 - w2)
        
        Vec3 a1_W = s1.pose_WB().rotate(axis1_B);
        Vec3 u2_W = s2.pose_WB().rotate(u2_B);
        Vec3 v2_W = s2.pose_WB().rotate(v2_B);

        Vec3 n1 = a1_W.cross(u2_W);
        Vec3 n2 = a1_W.cross(v2_W);

        Vec3 n1_dot = (w1.cross(a1_W)).cross(u2_W) + a1_W.cross(w2.cross(u2_W));
        Vec3 n2_dot = (w1.cross(a1_W)).cross(v2_W) + a1_W.cross(w2.cross(v2_W));

        gamma(3) = n1_dot.dot(w1 - w2);
        gamma(4) = n2_dot.dot(w1 - w2);
    }
};

} // namespace mbd