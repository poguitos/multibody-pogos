#include "mbd/vehicle/suspension/multilink.hpp"

#include <vector>

#include "mbd/vehicle/suspension/mass.hpp"

namespace mbd {

MultilinkCorner build_multilink_corner(
    kernel::System& sys,
    const MultilinkParams& p)
{
    using namespace kernel;
    MultilinkCorner ml;
    ml.params = p;
    // The joint's child-side frame sits at -wheel_center in the upright frame,
    // so the neutral configuration puts the upright at the wheel centre.
    ml.upright_body = sys.model.add_body(
        0, std::make_shared<FreeJointModel>(), Transform3::Identity(),
        Transform3::FromTranslation(-p.wheel_center),
        RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.05, 0.12)), "ML_upright");
    ml.free_joint_idx = ml.upright_body;

    for (std::size_t i = 0; i < 5; ++i) {
        ml.link_constraint_indices[i] = sys.constraints.size();
        sys.constraints.push_back(std::make_shared<Distance>(
            Marker{0, Transform3::FromTranslation(p.inner[i])},
            Marker{ml.upright_body, Transform3::FromTranslation(p.outer[i] - p.wheel_center)},
            (p.outer[i] - p.inner[i]).norm()));
    }

    ml.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_height_driver(ml.upright_body, Vec3::Zero(), 2, p.wheel_center.z()));
    return ml;
}

MultilinkCorner build_multilink_corner_dynamic(
    kernel::System& sys,
    BodyIndex chassis_body,
    const MultilinkParams& p,
    BodyIndex toe_link_body)
{
    using namespace kernel;
    if (toe_link_body < 0) toe_link_body = chassis_body;
    MultilinkCorner ml;
    ml.params = p;
    std::vector<Vec3> joints{Vec3::Zero()};   // the wheel centre, then the links' outer ends
    for (const Vec3& o : p.outer) joints.push_back(o - p.wheel_center);
    ml.upright_body = sys.model.add_body(
        chassis_body, std::make_shared<FreeJointModel>(), Transform3::Identity(),
        Transform3::FromTranslation(-p.wheel_center), lumped_inertia(p.upright_mass, joints), "dyn_ML_upright");
    ml.free_joint_idx = ml.upright_body;

    for (std::size_t i = 0; i < 5; ++i) {
        const BodyIndex inner_body = (i == 4) ? toe_link_body : chassis_body;
        ml.link_constraint_indices[i] = sys.constraints.size();
        sys.constraints.push_back(std::make_shared<Distance>(
            Marker{inner_body, Transform3::FromTranslation(p.inner[i])},
            Marker{ml.upright_body, Transform3::FromTranslation(p.outer[i] - p.wheel_center)},
            (p.outer[i] - p.inner[i]).norm()));
    }
    ml.bump_constraint_idx = 0;   // no bump driver in a dynamic corner
    return ml;
}

} // namespace mbd
