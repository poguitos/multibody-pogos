#include "mbd/vehicle/suspension/multilink.hpp"

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
        RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.12, 0.05)), "ML_upright");
    ml.free_joint_idx = ml.upright_body;

    for (std::size_t i = 0; i < 5; ++i) {
        ml.link_constraint_indices[i] = sys.constraints.size();
        sys.constraints.push_back(std::make_shared<Distance>(
            Marker{0, Transform3::FromTranslation(p.inner[i])},
            Marker{ml.upright_body, Transform3::FromTranslation(p.outer[i] - p.wheel_center)},
            (p.outer[i] - p.inner[i]).norm()));
    }

    ml.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_height_driver(ml.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    return ml;
}

} // namespace mbd
