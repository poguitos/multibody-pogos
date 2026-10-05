#include "mbd/vehicle/suspension/double_wishbone.hpp"

namespace mbd {

Mat3 detail::rotation_align_z_to(const Vec3& target)
{
    Vec3 z = target.normalized();
    Vec3 x;
    if (std::abs(z.dot(Vec3::UnitY())) < 0.9) {
        x = z.cross(Vec3::UnitY()).normalized();
    } else {
        x = z.cross(Vec3::UnitX()).normalized();
    }
    Vec3 y = z.cross(x).normalized();

    Mat3 R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return R;
}

DoubleWishboneCorner build_double_wishbone_corner(
    kernel::System& sys,
    const DoubleWishboneParams& p)
{
    using namespace kernel;
    DoubleWishboneCorner dwb;
    dwb.params = p;
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);
    const auto I_arm = RigidBodyInertia::from_solid_box(p.arm_mass, Vec3(0.02, 0.02, 0.2));
    const auto I_upright = RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.1, 0.05));
    const auto revolute = std::make_shared<RevoluteJointModel>();

    dwb.lca_body = sys.model.add_body(0, revolute, Transform3(R_arm, p.lca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "LCA");
    dwb.upright_body = sys.model.add_body(
        dwb.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I_upright, "upright");
    dwb.uca_body = sys.model.add_body(0, revolute, Transform3(R_arm, p.uca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "UCA");
    dwb.lca_joint_idx = dwb.lca_body;
    dwb.spherical_joint_idx = dwb.upright_body;
    dwb.uca_joint_idx = dwb.uca_body;

    // Upper ball joint: the UCA's outer point on the upright.
    dwb.coincident_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{dwb.uca_body, Transform3::FromTranslation(p.uca_outer - p.uca_pivot)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.uca_outer - p.wheel_center)}));

    // Tie rod.
    dwb.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{0, Transform3::FromTranslation(p.tierod_inner)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.tierod_outer - p.wheel_center)},
        (p.tierod_outer - p.tierod_inner).norm()));

    // Bump prescription: wheel-centre height = wheel_center.y() + t.
    dwb.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_height_driver(dwb.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    return dwb;
}

DoubleWishboneCorner build_double_wishbone_corner_dynamic(
    kernel::System& sys,
    BodyIndex chassis_body,
    const DoubleWishboneParams& p,
    BodyIndex tierod_body)
{
    using namespace kernel;
    if (tierod_body < 0) tierod_body = chassis_body;
    DoubleWishboneCorner dwb;
    dwb.params = p;
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);
    const auto I_arm = RigidBodyInertia::from_solid_box(p.arm_mass, Vec3(0.02, 0.02, 0.2));
    const auto I_upright = RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.1, 0.05));
    const auto revolute = std::make_shared<RevoluteJointModel>();

    // Arm frames: origin at the pivot, axes of the chassis at the reference.
    dwb.lca_body = sys.model.add_body(chassis_body, revolute, Transform3(R_arm, p.lca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "dyn_LCA");
    dwb.upright_body = sys.model.add_body(
        dwb.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I_upright, "dyn_upright");
    dwb.uca_body = sys.model.add_body(chassis_body, revolute, Transform3(R_arm, p.uca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "dyn_UCA");
    dwb.lca_joint_idx = dwb.lca_body;
    dwb.spherical_joint_idx = dwb.upright_body;
    dwb.uca_joint_idx = dwb.uca_body;

    dwb.coincident_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{dwb.uca_body, Transform3::FromTranslation(p.uca_outer - p.uca_pivot)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.uca_outer - p.wheel_center)}));

    dwb.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{tierod_body, Transform3::FromTranslation(p.tierod_inner)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.tierod_outer - p.wheel_center)},
        (p.tierod_outer - p.tierod_inner).norm()));

    dwb.bump_constraint_idx = 0;   // no bump driver in a dynamic corner
    return dwb;
}

} // namespace mbd
