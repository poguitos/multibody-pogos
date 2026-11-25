#include "mbd/vehicle/wheel.hpp"

namespace mbd {

RigidBodyInertia wheel_inertia(Real mass, Real radius)
{
    const Real diameter = 0.25 * mass * radius * radius;
    const Real axle = 0.5 * mass * radius * radius;
    return RigidBodyInertia(mass, Vec3::Zero(), Vec3(diameter, axle, diameter).asDiagonal());
}

BodyIndex add_wheel(kernel::Model& model, BodyIndex carrier, const Vec3& centre, Real mass, Real radius,
                    const std::string& name)
{
    // Rx(-pi/2) maps the joint's Z (its axis) to the carrier's Y; the same
    // rotation on the wheel's side keeps the wheel's axes the carrier's at
    // zero.
    const Mat3 R_spin = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    return model.add_body(carrier, std::make_shared<kernel::RevoluteJointModel>(), Transform3(R_spin, centre),
                          Transform3::FromRotation(R_spin), wheel_inertia(mass, radius), name);
}

void set_free_rolling(kernel::Simulator& sim, BodyIndex wheel, const FullTireForce& tire)
{
    sim.refresh();
    const RigidBodyState& s = sim.states()[static_cast<std::size_t>(wheel)];
    const Real spin = s.w_WB.dot(s.q_WB * Vec3::UnitY());
    sim.v(sim.system.model.idx_v[static_cast<std::size_t>(wheel)]) += tire.free_rolling_spin(sim.states()) - spin;
    sim.refresh();
}

} // namespace mbd
