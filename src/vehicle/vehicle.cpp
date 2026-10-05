#include "mbd/vehicle/vehicle.hpp"

namespace mbd {

std::pair<Real, Real> VehicleModel::ackermann_steering(Real delta) const
{
    if (std::abs(delta) < Real(1e-10)) return {Real(0.0), Real(0.0)};

    const Real L  = params.front_axle_x + params.rear_axle_x;
    const Real ht = params.half_track;
    const Real R  = L / std::tan(delta);

    const Real delta_FL = std::atan(L / (R - ht));
    const Real delta_FR = std::atan(L / (R + ht));

    return {delta_FL, delta_FR};
}

void VehicleModel::set_front_steering(Real delta)
{
    auto [d_FL, d_FR] = ackermann_steering(delta);
    tires[0]->steer_angle = d_FL;
    tires[1]->steer_angle = d_FR;
}

void VehicleModel::set_steering_angles(Real fl, Real fr, Real rl, Real rr)
{
    tires[0]->steer_angle = fl;
    tires[1]->steer_angle = fr;
    tires[2]->steer_angle = rl;
    tires[3]->steer_angle = rr;
}

void VehicleModel::clear_steering()
{
    for (auto* t : tires) {
        t->steer_angle = Real(0.0);
    }
}

VehicleModel build_simple_vehicle(kernel::System& sys,
                                  const VehicleParams& p)
{
    VehicleModel vm;
    vm.params = p;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);

    // Prismatic joint frames: Rx(pi/2) maps joint Z to chassis -Y, so a
    // positive coordinate moves the wheel down from its mount.
    const Mat3 R_susp = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();

    // --- Chassis (body 1) ---
    vm.chassis_body = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
        Transform3::Identity(),
        RigidBodyInertia::from_solid_box(p.chassis_mass, p.chassis_half_extents), "chassis");
    vm.chassis_joint_index = vm.chassis_body;

    // --- Wheel mount positions in chassis frame ---
    const std::array<Vec3, 4> mount_pos = {{
        Vec3( p.front_axle_x, 0.0,  p.half_track),  // FL
        Vec3( p.front_axle_x, 0.0, -p.half_track),  // FR
        Vec3(-p.rear_axle_x,  0.0,  p.half_track),  // RL
        Vec3(-p.rear_axle_x,  0.0, -p.half_track),  // RR
    }};

    const std::array<std::string, 4> names = {{"FL", "FR", "RL", "RR"}};
    const auto I_wheel = RigidBodyInertia::from_solid_box(p.wheel_mass, p.wheel_half_extents);
    const auto prismatic = std::make_shared<kernel::PrismaticJointModel>();

    for (std::size_t c = 0; c < 4; ++c) {
        // --- Wheel body on a prismatic joint from the chassis ---
        vm.wheel_bodies[c] = sys.model.add_body(
            vm.chassis_body, prismatic, Transform3(R_susp, mount_pos[c]),
            Transform3::FromRotation(R_susp), I_wheel, names[c]);
        vm.wheel_joint_indices[c] = vm.wheel_bodies[c];
        vm.susp_q[c] = sys.model.idx_q[vm.wheel_bodies[c]];
        vm.susp_v[c] = sys.model.idx_v[vm.wheel_bodies[c]];

        // --- Suspension spring-damper: chassis mount to wheel origin ---
        sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
            vm.chassis_body, vm.wheel_bodies[c],
            mount_pos[c], Vec3::Zero(),
            p.k_susp, p.c_susp, p.susp_rest_length));

        // --- Tire force ---
        auto tire = std::make_shared<FullTireForce>(
            vm.wheel_bodies[c],
            p.tire_free_radius,
            p.tire_k_z,
            p.tire_c_z,
            p.tire_params);
        vm.tires[c] = tire.get();
        sys.force_elements.push_back(std::move(tire));
    }

    return vm;
}

void set_vehicle_equilibrium(kernel::Simulator& sim, const VehicleModel& vm)
{
    const auto& p = vm.params;

    // Chassis: centered at equilibrium height, no rotation
    sim.q = sim.system.model.neutral_configuration();
    sim.q(vm.chassis_ty_idx()) = p.chassis_height_eq();

    // Wheels: each at equilibrium suspension extension
    for (std::size_t c = 0; c < 4; ++c) {
        sim.q(vm.susp_q[c]) = p.q_susp_eq();
    }

    sim.v.setZero();
    sim.refresh();
}

} // namespace mbd
