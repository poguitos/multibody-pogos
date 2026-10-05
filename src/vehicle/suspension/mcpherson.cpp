#include "mbd/vehicle/suspension/mcpherson.hpp"

namespace mbd {

McPhersonCorner build_mcpherson_corner(
    kernel::System& sys,
    const McPhersonParams& p)
{
    using namespace kernel;
    McPhersonCorner mc;
    mc.params = p;
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);
    const auto I_arm = RigidBodyInertia::from_solid_box(p.arm_mass, Vec3(0.02, 0.02, 0.2));
    const auto I_upright = RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.1, 0.05));

    mc.lca_body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(),
                                     Transform3(R_arm, p.lca_pivot),
                                     Transform3::FromRotation(R_arm), I_arm, "MC_LCA");
    mc.upright_body = sys.model.add_body(
        mc.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I_upright, "MC_upright");
    mc.lca_joint_idx = mc.lca_body;
    mc.spherical_joint_idx = mc.upright_body;

    // The strut's top mount stays on the strut axis, a line on the upright
    // through strut_lower (the Z axis of the marker).
    const Vec3 strut_axis = (p.strut_top_mount - p.strut_lower).normalized();
    mc.strut_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_on_line(
        Marker{mc.upright_body, Transform3(detail::rotation_align_z_to(strut_axis),
                                           p.strut_lower - p.wheel_center)},
        Marker{0, Transform3::FromTranslation(p.strut_top_mount)}));

    mc.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{0, Transform3::FromTranslation(p.tierod_inner)},
        Marker{mc.upright_body, Transform3::FromTranslation(p.tierod_outer - p.wheel_center)},
        (p.tierod_outer - p.tierod_inner).norm()));

    mc.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_height_driver(mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    return mc;
}

McPhersonCorner build_mcpherson_corner_dynamic(
    kernel::System& sys,
    BodyIndex chassis_body,
    const McPhersonParams& p,
    BodyIndex tierod_body)
{
    using namespace kernel;
    if (tierod_body < 0) tierod_body = chassis_body;
    McPhersonCorner mc;
    mc.params = p;
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);
    const auto I_arm = RigidBodyInertia::from_solid_box(p.arm_mass, Vec3(0.02, 0.02, 0.2));
    const auto I_upright = RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.1, 0.05));

    mc.lca_body = sys.model.add_body(chassis_body, std::make_shared<RevoluteJointModel>(),
                                     Transform3(R_arm, p.lca_pivot),
                                     Transform3::FromRotation(R_arm), I_arm, "dyn_MC_LCA");
    mc.upright_body = sys.model.add_body(
        mc.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I_upright, "dyn_MC_upright");
    mc.lca_joint_idx = mc.lca_body;
    mc.spherical_joint_idx = mc.upright_body;

    const Vec3 strut_axis = (p.strut_top_mount - p.strut_lower).normalized();
    mc.strut_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_on_line(
        Marker{mc.upright_body, Transform3(detail::rotation_align_z_to(strut_axis),
                                           p.strut_lower - p.wheel_center)},
        Marker{chassis_body, Transform3::FromTranslation(p.strut_top_mount)}));

    mc.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{tierod_body, Transform3::FromTranslation(p.tierod_inner)},
        Marker{mc.upright_body, Transform3::FromTranslation(p.tierod_outer - p.wheel_center)},
        (p.tierod_outer - p.tierod_inner).norm()));

    mc.bump_constraint_idx = 0;
    return mc;
}

} // namespace mbd
