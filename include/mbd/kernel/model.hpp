#pragma once

// Model and Data of the kinematics kernel (plan task 2.3, docs/kernel.md).
//
// Model holds everything that does not change during a simulation: the tree,
// the joints, where they sit on the bodies, the inertias and gravity. It can
// be shared, read-only, between threads.
//
// Data holds everything the algorithms compute. It is sized once for a Model,
// so the algorithms do not allocate memory. Each thread uses its own Data.

#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/model/rigid_body.hpp"
#include "mbd/kernel/joint_model.hpp"

namespace mbd::kernel {

class Model {
public:
    /// An empty model: ground only (body 0).
    Model();

    /// Add a body attached to `parent_body` (0 for ground) by `joint_model`.
    ///   x_pj          the parent-side joint frame, in the parent body frame;
    ///   x_cj          the child-side joint frame, in the new body's frame;
    ///   body_inertia  mass, centre of mass and rotational inertia, in the
    ///                 new body's frame.
    /// Returns the index of the new body (>= 1). Bodies are numbered in the
    /// order they are added, so a parent always has a smaller index.
    int add_body(int parent_body,
                 std::shared_ptr<const JointModel> joint_model,
                 const Transform3& x_pj,
                 const Transform3& x_cj,
                 const RigidBodyInertia& body_inertia,
                 std::string body_name = "");

    /// Number of bodies, ground included.
    int nbodies() const { return static_cast<int>(parent.size()); }

    /// Total numbers of coordinates and velocities.
    int nq{0};
    int nv{0};

    /// Gravitational acceleration in the world frame. The default follows
    /// ISO 8855 (Z up); set it for other conventions.
    Vec3 gravity{0.0, 0.0, -g_accel};

    // --- Per body (index 0 is ground, whose entries are unused) -------------

    std::vector<int> parent;                               ///< Parent body; -1 for ground
    std::vector<std::shared_ptr<const JointModel>> joint;  ///< Joint to the parent
    std::vector<Transform3> X_PJ;                          ///< Parent-side joint frame
    std::vector<Transform3> X_CJ;                          ///< Child-side joint frame
    std::vector<RigidBodyInertia> inertia;
    std::vector<int> idx_q;                                ///< First coordinate of the joint
    std::vector<int> idx_v;                                ///< First velocity of the joint
    std::vector<int> nqs;                                  ///< Coordinates of the joint
    std::vector<int> nvs;                                  ///< Velocities of the joint
    std::vector<std::string> name;

    // --- Derived once, when the body is added --------------------------------

    std::vector<Transform3> X_JC;    ///< X_CJ inverse
    std::vector<Mat6> X_CJ_motion;   ///< motion_matrix(X_CJ)
    std::vector<Mat6> I;             ///< Spatial inertia about the body origin

    /// The reference configuration: every joint at its neutral coordinates
    /// (identity quaternions, not zeros).
    VecX neutral_configuration() const;
};

/// Working storage for the algorithms, sized for one Model. Quantities in the
/// body frame unless stated otherwise.
struct Data {
    explicit Data(const Model& model);

    std::vector<JointData> joint;   ///< X_J, S, c of each joint, child-side joint frame
    std::vector<Mat6X> S;           ///< Motion subspace of each joint, in the body frame
    std::vector<Transform3> liMi;   ///< Body placement in the parent body frame
    std::vector<Transform3> oMi;    ///< Body placement in the world
    std::vector<Vec6> v;            ///< Body velocity
    std::vector<Vec6> a;            ///< Body acceleration
    std::vector<Vec6> a_gf;         ///< rnea, aba: body acceleration minus gravity
    std::vector<Vec6> c;            ///< Velocity-product acceleration of the joint
    std::vector<Vec6> f;            ///< Force transmitted by the joint, about the body origin
    std::vector<Mat6> Ic;           ///< Composite (CRBA) or articulated (ABA) inertia
    std::vector<Vec6> pA;           ///< Articulated bias force (ABA)
    std::vector<Mat6X> U;           ///< ABA: articulated inertia times S
    std::vector<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6>> Dinv;  ///< ABA
    std::vector<Eigen::Matrix<Real, Eigen::Dynamic, 1, 0, 6, 1>> u;                  ///< ABA

    MatX M;        ///< Mass matrix (crba)
    VecX tau;      ///< Generalized forces (rnea)
    VecX ddq;      ///< Accelerations (aba)
    VecX zero_v;   ///< Zero velocities, used when only positions are wanted
};

} // namespace mbd::kernel
