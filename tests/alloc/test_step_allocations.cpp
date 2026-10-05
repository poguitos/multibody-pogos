// No allocation per step (plan task 2.10): once a simulation is set up and has
// taken its first steps, stepping it allocates no memory. Every allocation
// through operator new is counted (alloc_counter.cpp); in a build with
// EIGEN_RUNTIME_NO_MALLOC, Eigen's own allocations, which use malloc, fail an
// assertion as well (the CI job kernel-no-malloc).

#include <catch2/catch_test_macros.hpp>

#include <memory>

#include <Eigen/Core>

#include "alloc/alloc_counter.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/drivetrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using namespace mbd;

namespace {

/// Allocations made during `steps` steps of 1 ms.
long allocations_in_steps(kernel::Simulator& sim, int steps)
{
#if defined(EIGEN_RUNTIME_NO_MALLOC)
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    mbd_test::start_counting_allocations();
    for (int k = 0; k < steps; ++k) sim.step(1e-3);
    const long n = mbd_test::stop_counting_allocations();
#if defined(EIGEN_RUNTIME_NO_MALLOC)
    Eigen::internal::set_is_malloc_allowed(true);
#endif
    return n;
}

} // namespace

TEST_CASE("Stepping the double-wishbone sedan allocates no memory", "[alloc][vehicle]")
{
    // Chassis, steering rack, four corners closed by constraints, springs,
    // dampers, tyres and the drivetrain: everything a step does.
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    dt.throttle = 0.3;
    vh.set_steering(0.02);
    sim.run(0.05, 1e-3);   // past the first steps, which may report warnings

    CHECK(allocations_in_steps(sim, 100) == 0);
}

TEST_CASE("Stepping a constrained mechanism allocates no memory", "[alloc][kernel]")
{
    // A planar four-bar closed in 3D (redundant equations) on a spring, with
    // both integrators.
    kernel::System sys;
    const auto revolute = std::make_shared<kernel::RevoluteJointModel>();
    auto link = [](Real mass, const Vec3& com) {
        RigidBodyInertia I = RigidBodyInertia::from_solid_box(mass, Vec3(0.05, 0.05, 0.05));
        I.com_B = com;
        return I;
    };
    const int crank = sys.model.add_body(0, revolute, Transform3(), Transform3(),
                                         link(1.0, Vec3(0.0, 0.5, 0.0)));
    const int coupler = sys.model.add_body(crank, revolute,
                                           Transform3::FromTranslation(Vec3(0.0, 1.0, 0.0)),
                                           Transform3(), link(2.0, Vec3(1.0, 0.0, 0.0)));
    const int rocker = sys.model.add_body(coupler, revolute,
                                          Transform3::FromTranslation(Vec3(2.0, 0.0, 0.0)),
                                          Transform3(), link(1.0, Vec3(0.0, -0.5, 0.0)));
    sys.constraints.push_back(kernel::revolute_closure(
        kernel::Marker{rocker, Transform3::FromTranslation(Vec3(0.0, -1.0, 0.0))},
        kernel::Marker{0, Transform3::FromTranslation(Vec3(2.0, 0.0, 0.0))}));
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, coupler, Vec3(1.0, 2.0, 0.0), Vec3(1.0, 0.0, 0.0), 50.0, 2.0, 0.5));

    for (const auto method : {kernel::Integrator::RK4, kernel::Integrator::SemiImplicitEuler}) {
        kernel::Simulator sim(sys);
        sim.method = method;
        sim.v(0) = 1.0;
        sim.initialize();
        sim.run(0.01, 1e-3);   // the redundancy is reported in the first step

        CHECK(allocations_in_steps(sim, 100) == 0);
    }
}
