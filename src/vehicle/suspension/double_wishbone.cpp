#include "mbd/vehicle/suspension/double_wishbone.hpp"

#include "mbd/vehicle/suspension/mass.hpp"

namespace mbd {

namespace {

// The parts' inertias (task 7.6, F9): each arm a rod from its pivot to its
// ball joint, in its frame (origin at the pivot, the chassis's axes at the
// reference); the upright's mass among its two ball joints, the tie rod's
// end and the wheel centre, in its frame (origin at the wheel centre).
struct DwbInertias {
    RigidBodyInertia lca, uca, upright;
};

DwbInertias dwb_inertias(const DoubleWishboneParams& p)
{
    return {rod_inertia(p.arm_mass, Vec3::Zero(), p.lca_outer - p.lca_pivot),
            rod_inertia(p.arm_mass, Vec3::Zero(), p.uca_outer - p.uca_pivot),
            lumped_inertia(p.upright_mass, {p.lca_outer - p.wheel_center, p.uca_outer - p.wheel_center,
                                            p.tierod_outer - p.wheel_center, Vec3::Zero()})};
}

} // namespace

Mat3 detail::rotation_align_z_to(const Vec3& target)
{
    Vec3 z = target.normalized();
    Vec3 x;
    if (std::abs(z.dot(Vec3::UnitZ())) < 0.9) {        // Z: up (ISO 8855)
        x = z.cross(Vec3::UnitZ()).normalized();
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
    const DwbInertias I = dwb_inertias(p);
    const auto revolute = std::make_shared<RevoluteJointModel>();

    dwb.lca_body = sys.model.add_body(0, revolute, Transform3(R_arm, p.lca_pivot),
                                      Transform3::FromRotation(R_arm), I.lca, "LCA");
    dwb.upright_body = sys.model.add_body(
        dwb.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I.upright, "upright");
    dwb.uca_body = sys.model.add_body(0, revolute, Transform3(R_arm, p.uca_pivot),
                                      Transform3::FromRotation(R_arm), I.uca, "UCA");
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

    // Bump prescription: wheel-centre height = wheel_center.z() + t.
    dwb.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_height_driver(dwb.upright_body, Vec3::Zero(), 2, p.wheel_center.z()));
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
    const DwbInertias I = dwb_inertias(p);
    const auto revolute = std::make_shared<RevoluteJointModel>();

    // Arm frames: origin at the pivot, axes of the chassis at the reference.
    dwb.lca_body = sys.model.add_body(chassis_body, revolute, Transform3(R_arm, p.lca_pivot),
                                      Transform3::FromRotation(R_arm), I.lca, "dyn_LCA");
    dwb.upright_body = sys.model.add_body(
        dwb.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I.upright, "dyn_upright");
    dwb.uca_body = sys.model.add_body(chassis_body, revolute, Transform3(R_arm, p.uca_pivot),
                                      Transform3::FromRotation(R_arm), I.uca, "dyn_UCA");
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
