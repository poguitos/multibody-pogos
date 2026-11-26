#include "mbd/vehicle/suspension/trailing_arm.hpp"

#include "mbd/vehicle/suspension/double_wishbone.hpp"   // detail::rotation_align_z_to
#include "mbd/vehicle/suspension/mass.hpp"

namespace mbd {

TrailingArmCorner build_trailing_arm_corner(kernel::System& sys, BodyIndex chassis_body, const TrailingArmParams& p)
{
    MBD_THROW_IF(!(p.axis.norm() > 0.0) || !((p.pivot - p.wheel_center).norm() > 0.0),
                 "MBD-V002: build_trailing_arm_corner: the hinge axis must not be zero, nor the pivot on the wheel "
                 "centre");
    TrailingArmCorner c;
    c.params = p;
    // The joint's Z on the hinge axis on both sides; the arm's origin at the
    // wheel centre, so the child-side joint frame sits at pivot - wheel_center.
    const Mat3 R = detail::rotation_align_z_to(p.axis);
    const Vec3 pivot_B = p.pivot - p.wheel_center;
    const RigidBodyInertia I = combine(rod_inertia(p.arm_mass, pivot_B, Vec3::Zero()),
                                       lumped_inertia(p.hub_mass, {Vec3::Zero()}));
    c.arm_body = sys.model.add_body(chassis_body, std::make_shared<kernel::RevoluteJointModel>(),
                                    Transform3(R, p.pivot), Transform3(R, pivot_B), I, "trailing_arm");
    return c;
}

} // namespace mbd
