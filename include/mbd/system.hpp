#pragma once

#include <vector>
#include <memory>
#include <string>

#include "mbd/core.hpp"
#include "mbd/rigid_body.hpp"
#include "mbd/force_element.hpp"
#include "mbd/dynamics.hpp"
#include "mbd/joint.hpp"

namespace mbd {

class Constraint;

/// Per-body metadata (name, topology).
struct BodyInfo {
    std::string name;
    BodyIndex parent_idx{kNoParent};
    int joint_idx{-1}; // index into MultibodySystem::joints, -1 for ground

    BodyInfo() = default;
    explicit BodyInfo(std::string n, BodyIndex parent = kNoParent, int j_idx = -1)
        : name(std::move(n)), parent_idx(parent), joint_idx(j_idx) {}
};

/// The central container for a multibody simulation.
///
/// Body 0 is always ground: identity pose, zero velocity, never integrated.
class MultibodySystem {
public:
    // --- Per-body parallel arrays (indexed by BodyIndex) --------------------
    std::vector<RigidBodyState>   states;
    std::vector<RigidBodyInertia> inertias;
    std::vector<RigidBodyForces>  forces;
    std::vector<BodyInfo>         body_infos;

    // --- Joints (tree topology) ---------------------------------------------
    std::vector<std::unique_ptr<Joint>> joints;

    // --- Generalized coordinates and velocities -----------------------------
    VecX q;      // joint coordinates (size = total DOFs)
    VecX q_dot;  // joint velocities  (size = total DOFs)
    int total_dof{0};

    // --- Force elements and constraints -------------------------------------
    std::vector<std::unique_ptr<ForceElement>>  force_elements;
    std::vector<std::shared_ptr<Constraint>>    constraints;

    /// Constructor: creates ground body at index 0.
    MultibodySystem()
    {
        states.push_back(RigidBodyState{});
        inertias.push_back(RigidBodyInertia{});
        forces.push_back(RigidBodyForces{});
        body_infos.push_back(BodyInfo{"ground", kNoParent, -1});
    }

    /// Add a rigid body. Returns its BodyIndex (always >= 1).
    BodyIndex add_body(const RigidBodyInertia& inertia,
                       const RigidBodyState& initial_state = RigidBodyState{},
                       const std::string& name = "",
                       BodyIndex parent = kGroundIndex)
    {
        states.push_back(initial_state);
        inertias.push_back(inertia);
        forces.push_back(RigidBodyForces{});
        body_infos.push_back(BodyInfo{name, parent, -1});
        return static_cast<BodyIndex>(states.size() - 1);
    }

    /// Add a joint connecting parent to child.
    /// The joint's parent_body_idx and child_body_idx must already be set.
    /// Returns the joint index. Resizes q and q_dot to accommodate new DOFs.
    int add_joint(std::unique_ptr<Joint> joint)
    {
        MBD_THROW_IF(joint->child_body_idx <= kGroundIndex ||
                     joint->child_body_idx >= body_count(),
                     "add_joint: invalid child body index");
        MBD_THROW_IF(joint->parent_body_idx < kGroundIndex ||
                     joint->parent_body_idx >= body_count(),
                     "add_joint: invalid parent body index");
        MBD_THROW_IF(joint->parent_body_idx >= joint->child_body_idx,
                     "add_joint: parent index must be < child index (topological order)");

        int j_idx = static_cast<int>(joints.size());
        int ndof  = joint->num_dof();

        joint->q_offset = total_dof;
        total_dof += ndof;

        // Resize q and q_dot, preserving existing values
        VecX q_new = VecX::Zero(total_dof);
        VecX qd_new = VecX::Zero(total_dof);
        if (q.size() > 0) {
            q_new.head(q.size()) = q;
            qd_new.head(q_dot.size()) = q_dot;
        }
        q = q_new;
        q_dot = qd_new;

        // Link body to its joint
        body_infos[joint->child_body_idx].parent_idx = joint->parent_body_idx;
        body_infos[joint->child_body_idx].joint_idx  = j_idx;

        joints.push_back(std::move(joint));
        return j_idx;
    }

    /// Get the segment of q corresponding to joint j.
    Eigen::VectorBlock<VecX> joint_q(int joint_idx)
    {
        auto& j = *joints[joint_idx];
        return q.segment(j.q_offset, j.num_dof());
    }

    Eigen::VectorBlock<const VecX> joint_q(int joint_idx) const
    {
        const auto& j = *joints[joint_idx];
        return q.segment(j.q_offset, j.num_dof());
    }

    /// Get the segment of q_dot corresponding to joint j.
    Eigen::VectorBlock<VecX> joint_q_dot(int joint_idx)
    {
        auto& j = *joints[joint_idx];
        return q_dot.segment(j.q_offset, j.num_dof());
    }

    Eigen::VectorBlock<const VecX> joint_q_dot(int joint_idx) const
    {
        const auto& j = *joints[joint_idx];
        return q_dot.segment(j.q_offset, j.num_dof());
    }

    // --- Queries ------------------------------------------------------------

    BodyIndex body_count() const { return static_cast<BodyIndex>(states.size()); }
    bool is_ground(BodyIndex idx) const { return idx == kGroundIndex; }
    int joint_count() const { return static_cast<int>(joints.size()); }

    void clear_forces()
    {
        for (auto& f : forces) {
            f.f_W.setZero();
            f.tau_W.setZero();
        }
    }

    void apply_force_elements()
    {
        for (const auto& fe : force_elements) {
            fe->apply(states, forces);
        }
    }

    // --- Forward Kinematics -------------------------------------------------

    /// Compute world-frame poses of all bodies from joint coordinates q.
    /// Traverses bodies in index order (which must be topological order).
    /// Updates states[i].p_WB and states[i].q_WB for all bodies.
    void compute_forward_kinematics()
    {
        // Ground is always identity
        states[kGroundIndex].p_WB = Vec3::Zero();
        states[kGroundIndex].q_WB = Quat::Identity();

        for (BodyIndex i = 1; i < body_count(); ++i) {
            const auto& info = body_infos[i];
            MBD_THROW_IF(info.joint_idx < 0,
                "compute_forward_kinematics: body has no joint");

            const auto& joint = *joints[info.joint_idx];
            const VecX q_j = joint_q(info.joint_idx);

            // Parent world pose
            const Transform3 X_WP = states[info.parent_idx].pose_WB();

            // Parent-to-child transform from joint coordinates
            const Transform3 X_PC = joint.parent_to_child_transform(q_j);

            // Child world pose
            const Transform3 X_WC = X_WP * X_PC;

            states[i].p_WB = X_WC.p;
            states[i].q_WB = X_WC.q;
        }
    }

    /// Compute world-frame velocities of all bodies from q and q_dot.
    /// Must be called after compute_forward_kinematics().
    /// Updates states[i].v_WB and states[i].w_WB for all bodies.
    ///
    /// Algorithm (for each body i with parent p, connected by joint j):
    ///   1. Compute velocity of the parent-side joint attachment point
    ///      (as a point rigidly attached to the parent body).
    ///   2. Add joint-relative linear velocity to get the moving joint
    ///      frame velocity.
    ///   3. Compute child angular velocity = parent angular velocity +
    ///      joint relative angular velocity.
    ///   4. Propagate from moving joint frame origin to child body origin
    ///      using the child's angular velocity.
    void compute_forward_velocities()
    {
        states[kGroundIndex].v_WB = Vec3::Zero();
        states[kGroundIndex].w_WB = Vec3::Zero();

        for (BodyIndex i = 1; i < body_count(); ++i) {
            const auto& info = body_infos[i];
            const auto& joint = *joints[info.joint_idx];

            const VecX q_j  = joint_q(info.joint_idx);
            const VecX qd_j = joint_q_dot(info.joint_idx);

            const Vec3& w_parent = states[info.parent_idx].w_WB;
            const Vec3& v_parent = states[info.parent_idx].v_WB;

            // --- Joint relative velocity in joint frame ---
            const auto S_joint = joint.motion_subspace(q_j);
            const Vec6 V_rel_joint = (joint.num_dof() > 0)
                ? Vec6(S_joint * qd_j)
                : Vec6::Zero();

            const Vec3 omega_rel_J = V_rel_joint.head<3>();
            const Vec3 v_rel_J     = V_rel_joint.tail<3>();

            // --- Rotate joint-frame quantities to world frame ---
            const Transform3 X_WP = states[info.parent_idx].pose_WB();
            const Transform3 X_J  = joint.joint_transform(q_j);
            const Quat q_WJ = X_WP.q * joint.X_PJ.q * X_J.q;

            const Vec3 omega_rel_W = q_WJ * omega_rel_J;
            const Vec3 v_rel_W     = q_WJ * v_rel_J;

            // --- Step 1: velocity of parent-side joint attachment ---
            const Vec3 p_J_parent_W = X_WP.apply(joint.X_PJ.p);
            const Vec3 r_parent_to_J = p_J_parent_W - states[info.parent_idx].p_WB;
            const Vec3 v_J_W = v_parent + w_parent.cross(r_parent_to_J);

            // --- Step 2: velocity of moving joint frame origin ---
            // (adds joint-relative linear velocity, e.g. prismatic sliding)
            const Vec3 v_Jmoving_W = v_J_W + v_rel_W;

            // --- Step 3: child angular velocity ---
            states[i].w_WB = w_parent + omega_rel_W;

            // --- Step 4: propagate from joint to child body origin ---
            // Moving joint frame origin in world (computed from child side):
            //   X_W_Jmoving = X_WC * X_CJ, so p_Jmoving = X_WC.apply(X_CJ.p)
            const Vec3 p_Jmoving_W = states[i].pose_WB().apply(joint.X_CJ.p);
            const Vec3 r_J_to_child = states[i].p_WB - p_Jmoving_W;

            states[i].v_WB = v_Jmoving_W + states[i].w_WB.cross(r_J_to_child);
        }
    }

    /// Convenience: compute both pose and velocity FK.
    void compute_kinematics()
    {
        compute_forward_kinematics();
        compute_forward_velocities();
    }
};

} // namespace mbd