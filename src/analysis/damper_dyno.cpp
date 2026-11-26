#include "mbd/analysis/damper_dyno.hpp"

#include <cmath>
#include <string>

#include "mbd/kernel/constraints.hpp"

namespace mbd {

Real DynoRecord::energy(int cycle) const
{
    const std::size_t n = static_cast<std::size_t>(steps_per_cycle);
    const std::size_t first = static_cast<std::size_t>(cycle) * n;
    MBD_THROW_IF(cycle < 0 || steps_per_cycle < 1 || first + n >= stroke.size(),
                 "MBD-A030: DynoRecord::energy: no cycle " + std::to_string(cycle) + " in the record");
    Real e = 0.0;
    for (std::size_t i = first; i < first + n; ++i) {
        e += 0.5 * (force[i] * velocity[i] + force[i + 1] * velocity[i + 1]) * (time[i + 1] - time[i]);
    }
    return e;
}

DynoRecord run_damper_dyno(const DynoElement& make, const DynoSettings& s)
{
    MBD_THROW_IF(!make, "MBD-A030: run_damper_dyno: no element to test");
    const bool finite = std::isfinite(s.amplitude) && std::isfinite(s.frequency) && std::isfinite(s.length)
                     && s.axis.allFinite();
    MBD_THROW_IF(!finite || !(s.amplitude > 0.0) || !(s.frequency > 0.0) || !(s.length >= 0.0)
                     || !(s.axis.norm() > 0.0) || s.cycles < 1 || s.steps_per_cycle < 4,
                 "MBD-A030: run_damper_dyno: the amplitude and frequency must be > 0, the length >= 0, the axis "
                 "not zero, cycles >= 1 and steps_per_cycle >= 4");
    const Vec3 axis = s.axis.normalized();

    // The driven end: a prismatic joint along the axis (the joint's Z turned
    // onto it on both sides, so that the body keeps the world's axes).
    kernel::System sys;
    sys.model.gravity = Vec3::Zero();
    const Mat3 R = Eigen::Quaternion<Real>::FromTwoVectors(Vec3::UnitZ(), axis).toRotationMatrix();
    const BodyIndex driven = sys.model.add_body(
        0, std::make_shared<kernel::PrismaticJointModel>(), Transform3(R, s.length * axis),
        Transform3::FromRotation(R), RigidBodyInertia::from_solid_box(1.0, Vec3(0.01, 0.01, 0.01)),
        "dyno_rod");
    const Real U = s.amplitude;
    const Real w = 2.0 * pi * s.frequency;
    sys.constraints.push_back(std::make_shared<kernel::JointDriver>(
        sys.model, driven,
        kernel::TimeFunction([U, w](Real t) { return U * std::sin(w * t); },
                             [U, w](Real t) { return U * w * std::cos(w * t); },
                             [U, w](Real t) { return -U * w * w * std::sin(w * t); })));
    const std::shared_ptr<ForceElement> element = make(0, driven);
    MBD_THROW_IF(!element, "MBD-A030: run_damper_dyno: the element maker returned no element");
    sys.force_elements.push_back(element);

    kernel::Simulator sim(sys);
    sim.method = s.method;
    sim.step_control = s.step_control;
    sim.initialize();

    const int iq = sys.model.idx_q[static_cast<std::size_t>(driven)];
    const int iv = sys.model.idx_v[static_cast<std::size_t>(driven)];
    const int steps = s.cycles * s.steps_per_cycle;
    const Real h = 1.0 / (s.frequency * s.steps_per_cycle);

    DynoRecord r;
    r.steps_per_cycle = s.steps_per_cycle;
    const std::size_t n = static_cast<std::size_t>(steps) + 1;
    r.time.reserve(n);
    r.stroke.reserve(n);
    r.velocity.reserve(n);
    r.force.reserve(n);
    std::vector<RigidBodyForces> f(static_cast<std::size_t>(sys.model.nbodies()));
    auto sample = [&] {
        // The forces at the step's end state: an evaluation there hands the
        // state elements their states at that state, then the element alone
        // is applied.
        sim.acceleration(sim.q, sim.v, sim.x, sim.time);
        for (auto& b : f) b = RigidBodyForces{};
        element->apply(sim.states(), f);
        r.time.push_back(sim.time);
        r.stroke.push_back(sim.q(iq));
        r.velocity.push_back(sim.v(iv));
        r.force.push_back(-f[static_cast<std::size_t>(driven)].f_W.dot(axis));
    };
    sample();
    for (int i = 0; i < steps; ++i) {
        sim.step(h);
        sample();
    }
    return r;
}

} // namespace mbd
