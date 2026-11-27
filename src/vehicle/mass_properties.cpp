// A vehicle's mass properties (plan task 7.11, decision D72): the chassis
// body from its inputs, and the roll-up of the built car.

#include "mbd/vehicle/mass_properties.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/vehicle/suspension/mass.hpp"

namespace mbd {

namespace {

/// Throw MBD-V008 when `J` is no real body's rotational inertia.
void check_inertia(const Mat3& J, const std::string& what)
{
    Vec3 m = Vec3::Zero();
    const InertiaFault fault = inertia_fault(J, &m);
    MBD_THROW_IF(fault == InertiaFault::NotSymmetric,
                 "MBD-V008: chassis_inertia: " + what + ": its rotational inertia is not symmetric");
    MBD_THROW_IF(fault == InertiaFault::NegativeMoment,
                 "MBD-V008: chassis_inertia: " + what + ": its rotational inertia has a negative principal moment ("
                     + std::to_string(m(0)) + " kg m^2)");
    MBD_THROW_IF(fault == InertiaFault::TriangleInequality,
                 "MBD-V008: chassis_inertia: " + what + ": its principal moments of inertia (" + std::to_string(m(0))
                     + ", " + std::to_string(m(1)) + ", " + std::to_string(m(2))
                     + " kg m^2) break the triangle inequality: no real body has them");
}

const char* const corner_names[4] = {"FL", "FR", "RL", "RR"};

} // namespace

RigidBodyInertia chassis_inertia(const ChassisConfig& c)
{
    MBD_THROW_IF(!(c.mass > 0.0) || !std::isfinite(c.mass) || !c.com.allFinite(),
                 "MBD-V008: chassis_inertia: the chassis's mass must be positive and finite, and its centre of mass "
                 "finite");
    RigidBodyInertia body;
    if (c.inertia) {
        MBD_THROW_IF(!c.inertia->allFinite(), "MBD-V008: chassis_inertia: the chassis's inertia is not finite");
        check_inertia(*c.inertia, "the chassis");
        body = RigidBodyInertia(c.mass, c.com, *c.inertia);
    } else {
        body = RigidBodyInertia::from_solid_box(c.mass, c.half_extents);
        body.com_B = c.com;
    }
    for (const LumpedMass& l : c.lumped_masses) {
        const std::string what = "the lumped mass '" + l.name + "'";
        MBD_THROW_IF(!(l.mass >= 0.0) || !std::isfinite(l.mass) || !l.position.allFinite() || !l.inertia.allFinite(),
                     "MBD-V008: chassis_inertia: " + what
                         + ": its mass must be finite and not negative, its position and inertia finite");
        check_inertia(l.inertia, what);
        body = combine(body, RigidBodyInertia(l.mass, l.position, l.inertia));
    }
    return body;
}

MassReport mass_report(const kernel::Simulator& sim, const VehicleHandle& vh)
{
    const kernel::Model& model = sim.system.model;
    const kernel::Data& data = sim.data();
    for (int c = 0; c < 4; ++c) {
        MBD_THROW_IF(vh.corners[static_cast<std::size_t>(c)].tire == nullptr,
                     std::string("MBD-V008: mass_report: corner ") + corner_names[c] + " has no tyre");
    }

    // The chassis frame: a world point p is R^T (p - o) in it.
    const RigidBodyState chassis = kernel::body_state(data, vh.chassis_body);
    const Mat3 Rt = chassis.q_WB.toRotationMatrix().transpose();
    auto in_chassis = [&](const Vec3& p_W) { return Vec3(Rt * (p_W - chassis.p_WB)); };

    MassReport r;
    r.name = vh.tmpl.name;

    // Each body's centre of mass and inertia in the chassis frame; the total
    // and the centre of mass first, then the inertia about that point (no
    // large terms cancelling).
    const int n = model.nbodies();
    std::vector<Vec3> com(static_cast<std::size_t>(n), Vec3::Zero());
    Vec3 first_moment = Vec3::Zero();
    for (int b = 1; b < n; ++b) {
        const RigidBodyInertia& I = model.inertia[static_cast<std::size_t>(b)];
        const RigidBodyState s = kernel::body_state(data, b);
        com[static_cast<std::size_t>(b)] = in_chassis(s.p_WB + s.q_WB * I.com_B);
        r.total_mass += I.mass;
        first_moment += I.mass * com[static_cast<std::size_t>(b)];
    }
    r.centre_of_mass = first_moment / r.total_mass;
    for (int b = 1; b < n; ++b) {
        const RigidBodyInertia& I = model.inertia[static_cast<std::size_t>(b)];
        const Mat3 R = Rt * kernel::body_state(data, b).q_WB.toRotationMatrix();
        r.inertia += R * I.I_com_B * R.transpose()
                   + point_inertia(I.mass, com[static_cast<std::size_t>(b)] - r.centre_of_mass);
    }

    // The height above the road beneath the centre of mass.
    const std::shared_ptr<const Road>& road = vh.corners[0].tire->road;
    const Vec3 cg_W = chassis.p_WB + chassis.q_WB * r.centre_of_mass;
    const RoadPoint ground = road->at(cg_W.x(), cg_W.y());
    r.cg_height = (cg_W - Vec3(cg_W.x(), cg_W.y(), ground.height)).dot(ground.normal);

    // Positions along the road beneath the centre of mass, from that point:
    // forward (the chassis's X laid on the road) and to the left.
    const Vec3 x_W = chassis.q_WB * Vec3::UnitX();
    const Vec3 forward = (x_W - x_W.dot(ground.normal) * ground.normal).normalized();
    const Vec3 left = ground.normal.cross(forward);

    // The contacts, the unsprung masses, and where the sprung mass's weight
    // acts: the whole car's first moment about the centre of mass (zero)
    // less each unsprung mass's at its contact.
    std::array<Vec3, 4> contact_W{};
    Real sprung_forward = 0.0, sprung_left = 0.0;
    r.sprung_mass = r.total_mass;
    for (std::size_t c = 0; c < 4; ++c) {
        const CornerHandle& corner = vh.corners[c];
        const RigidBodyState w = kernel::body_state(data, corner.wheel_body);
        contact_W[c] = wheel_contact(corner.tire->road, w.p_WB, w.q_WB * Vec3::UnitY(), corner.tire->R_free).point;
        r.contact[c] = in_chassis(contact_W[c]);
        r.unsprung[c] = corner.unsprung_mass;
        r.sprung_mass -= corner.unsprung_mass;
        sprung_forward -= corner.unsprung_mass * (contact_W[c] - cg_W).dot(forward);
        sprung_left -= corner.unsprung_mass * (contact_W[c] - cg_W).dot(left);
    }
    sprung_forward /= r.sprung_mass;
    sprung_left /= r.sprung_mass;

    // The sprung mass's split: between the axles, then side to side.
    auto ahead = [&](std::size_t c) { return (contact_W[c] - cg_W).dot(forward); };
    auto aside = [&](std::size_t c) { return (contact_W[c] - cg_W).dot(left); };
    const Real front = detail::lever_share(r.sprung_mass, sprung_forward, 0.5 * (ahead(0) + ahead(1)),
                                           0.5 * (ahead(2) + ahead(3)));
    const std::array<Real, 2> axle{{front, r.sprung_mass - front}};
    for (std::size_t a = 0; a < 2; ++a) {
        r.sprung[2 * a] = detail::lever_share(axle[a], sprung_left, aside(2 * a), aside(2 * a + 1));
        r.sprung[2 * a + 1] = axle[a] - r.sprung[2 * a];
    }
    return r;
}

std::ostream& operator<<(std::ostream& os, const MassReport& r)
{
    char line[160];
    auto put = [&](const char* format, auto... values) {
        std::snprintf(line, sizeof line, format, values...);
        os << line << '\n';
    };
    os << "Mass properties of " << r.name << " (chassis frame: X forward, Y left, Z up)\n";
    put("  total mass                         %10.3f kg", r.total_mass);
    put("  centre of mass                 x %9.4f  y %9.4f  z %9.4f m", r.centre_of_mass.x(), r.centre_of_mass.y(),
        r.centre_of_mass.z());
    put("  its height above the road          %10.4f m", r.cg_height);
    put("  inertia about it, roll (Ixx)       %10.2f kg m^2", r.roll_inertia());
    put("                    pitch (Iyy)      %10.2f kg m^2", r.pitch_inertia());
    put("                    yaw (Izz)        %10.2f kg m^2", r.yaw_inertia());
    put("  products of inertia      Ixy %9.3f  Ixz %9.3f  Iyz %9.3f kg m^2", r.inertia(0, 1), r.inertia(0, 2),
        r.inertia(1, 2));
    put("  sprung mass                        %10.3f kg", r.sprung_mass);
    os << "  corner    unsprung      sprung       total   [kg]   contact x, y [m]\n";
    for (int c = 0; c < 4; ++c) {
        const std::size_t k = static_cast<std::size_t>(c);
        put("  %-6s %11.3f %11.3f %11.3f          %8.4f %8.4f", corner_names[c], r.unsprung[k], r.sprung[k],
            r.corner_mass(c), r.contact[k].x(), r.contact[k].y());
    }
    return os;
}

} // namespace mbd
