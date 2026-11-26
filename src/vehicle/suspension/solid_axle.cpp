#include "mbd/vehicle/suspension/solid_axle.hpp"

#include <utility>

#include "mbd/kernel/constraints.hpp"
#include "mbd/vehicle/suspension/mass.hpp"

namespace mbd {

SolidAxle build_solid_axle(kernel::System& sys, BodyIndex chassis_body, const SolidAxleParams& p)
{
    using namespace kernel;
    MBD_THROW_IF(!(p.half_track > 0.0), "MBD-V002: build_solid_axle: the half track must be > 0");
    SolidAxle a;
    a.params = p;
    // The axle's frame at its centre: the free joint's child-side frame at
    // -axle_center, so that the neutral configuration puts it there.
    const Vec3 left(0.0, p.half_track, 0.0);
    a.axle_body = sys.model.add_body(chassis_body, std::make_shared<FreeJointModel>(), Transform3::Identity(),
                                     Transform3::FromTranslation(-p.axle_center),
                                     rod_inertia(p.axle_mass, left, -left), "solid_axle");

    const std::array<std::pair<Vec3, Vec3>, 4> links{{{p.lower_chassis[0], p.lower_axle[0]},
                                                       {p.lower_chassis[1], p.lower_axle[1]},
                                                       {p.upper_chassis, p.upper_axle},
                                                       {p.panhard_chassis, p.panhard_axle}}};
    for (std::size_t i = 0; i < 4; ++i) {
        const Real length = (links[i].second - links[i].first).norm();
        MBD_THROW_IF(!(length > 0.0), "MBD-V002: build_solid_axle: a link has zero length");
        a.link_constraint_indices[i] = sys.constraints.size();
        sys.constraints.push_back(std::make_shared<Distance>(
            Marker{chassis_body, Transform3::FromTranslation(links[i].first)},
            Marker{a.axle_body, Transform3::FromTranslation(links[i].second - p.axle_center)}, length));
    }
    return a;
}

} // namespace mbd
