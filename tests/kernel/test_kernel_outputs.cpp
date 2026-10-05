// Output requests (plan task 3.3): joint reactions, constraint forces from the
// multipliers, energies of the force elements, and the recorder. Checked
// against the pendulum's pivot force in closed form, against derivatives of
// the potentials, and against conservation of energy.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/analysis/recorder.hpp"
#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Real m_bob = 2.0, L_bob = 0.9;

RigidBodyInertia bob()
{
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(m_bob, Vec3(0.05, 0.04, 0.03));
    I.com_B = Vec3(0.0, 0.0, -L_bob);
    return I;
}

/// Frames whose Z axis is the world's Y.
Transform3 z_along_y()
{
    return Transform3(Quat(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX())), Vec3::Zero());
}

/// The force a hinge about the world Y axis through the origin exerts on the
/// bob at angle theta and rate w: m a - m g, with the centre of mass at
/// r = Ry(theta) (0, 0, -L) and a = alpha x r + w x (w x r),
/// alpha = -(m g L / I_p) sin theta.
Vec3 pivot_force(Real theta, Real w)
{
    const Real I_p = bob().I_com_B(1, 1) + m_bob * L_bob * L_bob;
    const Real alpha = -(m_bob * g_accel * L_bob / I_p) * std::sin(theta);
    const Vec3 r(-L_bob * std::sin(theta), 0.0, -L_bob * std::cos(theta));
    const Vec3 omega(0.0, w, 0.0);
    const Vec3 a = Vec3(0.0, alpha, 0.0).cross(r) + omega.cross(omega.cross(r));
    return m_bob * a - m_bob * Vec3(0.0, 0.0, -g_accel);
}

bool close(Real analytic, Real numeric)
{
    return std::abs(analytic - numeric) <= 1e-6 * std::max(1.0, std::abs(numeric));
}

} // namespace

TEST_CASE("Outputs: a pendulum's hinge reaction is m a - m g, with no moment", "[kernel][outputs]")
{
    // The bob's axes are principal and its centre of mass is on the arm, so the
    // hinge carries no moment about the pivot: gravity's moment is exactly what
    // turns the bob.
    System sys;
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_along_y(), z_along_y(), bob(), "bob");
    Simulator sim(sys);
    sim.q(0) = 1.0;
    sim.initialize();

    Real worst_force = 0.0, worst_moment = 0.0;
    for (int k = 0; k < 6; ++k) {
        sim.run(0.137, 1e-3);   // states spread over a swing
        const Loads loads = compute_loads(sim);
        const Quat R_WJ = sim.data().oMi[1].q * sys.model.X_CJ[1].q;   // joint frame to world
        const Vec6& f = loads.joint_reaction[1];
        worst_force = std::max(worst_force, (R_WJ * Vec3(f.tail<3>()) - pivot_force(sim.q(0), sim.v(0))).norm());
        worst_moment = std::max(worst_moment, f.head<3>().norm());
    }
    CHECK(worst_force < 1e-9 * m_bob * g_accel);
    CHECK(worst_moment < 1e-9 * m_bob * g_accel * L_bob);
}

TEST_CASE("Outputs: a revolute closure's multipliers give the same pivot force", "[kernel][outputs]")
{
    // The same bob on a free joint, held at the pivot by a revolute closure:
    // the constraint's wrench on the bob, from its multipliers, is the hinge's
    // force, and its moment about the pivot (the world origin) is zero.
    System sys;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), bob(), "bob");
    sys.constraints.push_back(revolute_closure(Marker{1, z_along_y()}, Marker{0, z_along_y()}));
    Simulator sim(sys);
    sim.q.segment<4>(3) = Quat(Eigen::AngleAxisd(1.0, Vec3::UnitY())).coeffs();
    sim.initialize();

    Real worst_force = 0.0, worst_moment = 0.0;
    for (int k = 0; k < 6; ++k) {
        sim.run(0.137, 1e-3);
        const Loads loads = compute_loads(sim);
        const Real theta = 2.0 * std::atan2(sim.q(4), sim.q(6));   // quaternion (x, y, z, w) about Y
        const Real w = sim.states()[1].w_WB.y();
        worst_force = std::max(worst_force, (Vec3(loads.constraint_W[1].tail<3>()) - pivot_force(theta, w)).norm());
        worst_moment = std::max(worst_moment, loads.constraint_W[1].head<3>().norm());
    }
    CHECK(worst_force < 1e-8 * m_bob * g_accel);
    CHECK(worst_moment < 1e-8 * m_bob * g_accel * L_bob);
}

TEST_CASE("Outputs: the force elements' potential energies give their forces", "[kernel][outputs]")
{
    // -dV/dx is the conservative part of the force (central differences,
    // h = 1e-6: truncation h^2 f'' / 6 below 1e-5 N, round-off 1e-7 N), and a
    // curve's integral has the curve as its derivative.
    const Real h = 1e-6;
    const Curve table = Curve::table({-0.1, 0.0, 0.05, 0.1}, {-2500.0, 0.0, 1400.0, 3600.0});
    for (const Real b : {-0.17, -0.043, 0.021, 0.077, 0.13}) {
        CHECK(close(table.value(b), (table.integral(-0.2, b + h) - table.integral(-0.2, b - h)) / (2.0 * h)));
    }
    CHECK(std::abs(Curve::linear(3.0, 1.0).integral(0.0, 2.0) - 8.0) < 1e-14);   // x + 1.5 x^2

    SpringDamperParams sp;
    sp.free_length = 0.35;
    sp.spring = table;
    sp.preload = 500.0;
    sp.bump_clearance = 0.06;
    sp.bump_stop = Curve::table({0.0, 0.01, 0.02}, {0.0, 2000.0, 8000.0});
    sp.rebound_clearance = 0.05;
    sp.rebound_stop = Curve::linear(5e5);
    const SpringDamper sd(0, 1, Vec3::Zero(), Vec3::Zero(), sp);
    for (const Real L : {0.2213, 0.2731, 0.3311, 0.4117, 0.4403}) {
        INFO("L = " << L);
        CHECK(close(sd.law(L, 0.0).force, -(sd.potential_energy(L + h) - sd.potential_energy(L - h)) / (2.0 * h)));
    }

    System sys;
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), bob(), "arm");
    JointCoordinateForceParams jp;
    jp.reference = 0.02;
    jp.spring = Curve::table({-0.2, 0.0, 0.1, 0.2}, {-900.0, 0.0, 300.0, 1100.0});
    jp.preload = 15.0;
    jp.lower_limit = -0.15;
    jp.upper_limit = 0.18;
    jp.limit_stiffness = 1e5;
    const JointCoordinateForce jf(sys.model, 1, jp);
    for (const Real q : {-0.1731, -0.0412, 0.0533, 0.1902}) {
        INFO("q = " << q);
        CHECK(close(jf.law(q, 0.0).force, -(jf.potential_energy(q + h) - jf.potential_energy(q - h)) / (2.0 * h)));
    }

    RotationalSpringDamperParams rp;
    rp.reference = 0.1;
    rp.spring = Curve::table({-1.0, 0.0, 0.5, 1.0}, {-300.0, 0.0, 120.0, 400.0});
    rp.preload = 4.0;
    const RotationalSpringDamper rsd(0, Transform3(), 1, Transform3(), rp);
    for (const Real a : {-0.73, 0.37, 0.88}) {
        CHECK(close(rsd.law(a, 0.0).force, -(rsd.potential_energy(a + h) - rsd.potential_energy(a - h)) / (2.0 * h)));
    }
}

TEST_CASE("Outputs: energy is conserved with springs and preloads", "[kernel][outputs]")
{
    // The bob swinging under gravity, a torsion spring with preload in its
    // hinge, and a spring with preload from its centre of mass to a ground
    // point: all conservative, so kinetic plus gravitational plus elastic
    // energy stays constant. RK4 at 0.1 ms on motions of a few rad/s drifts
    // by about (w dt)^4 w t, below 1e-11 of the energy.
    System sys;
    const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_along_y(), z_along_y(),
                                     bob(), "bob");
    JointCoordinateForceParams jp;
    jp.spring = Curve::linear(25.0);
    jp.preload = 3.0;
    const auto jf = std::make_shared<JointCoordinateForce>(sys.model, b, jp);
    sys.joint_forces.push_back(jf);
    SpringDamperParams sp;
    sp.free_length = 1.2;
    sp.spring = Curve::linear(150.0);
    sp.preload = 20.0;
    const Vec3 anchor(0.8, 0.0, -1.0), attach(0.0, 0.0, -L_bob);
    const auto sd = std::make_shared<SpringDamper>(0, b, anchor, attach, sp);
    sys.force_elements.push_back(sd);

    Simulator sim(sys);
    sim.q(0) = 0.6;
    sim.initialize();
    auto energy = [&] {
        const Vec3 p = sim.states()[1].pose_WB().apply(attach);
        return kinetic_energy(sys.model, sim.data()) + potential_energy(sys.model, sim.data())
             + jf->potential_energy(sim.q(0)) + sd->potential_energy((p - anchor).norm());
    };
    const Real E0 = energy();
    Real worst = 0.0;
    for (int s = 0; s < 20000; ++s) {
        sim.step(1e-4);
        worst = std::max(worst, std::abs(energy() - E0));
    }
    CHECK(worst < 1e-9 * std::abs(E0));
}

TEST_CASE("Outputs: the recorder samples named channels and writes them", "[kernel][outputs]")
{
    Recorder rec;
    Real x = 0.0;
    rec.add("x", [&] { return x; });
    rec.add("x_squared", [&] { return x * x; });
    CHECK_THROWS_WITH(rec.add("x", [] { return 0.0; }), ContainsSubstring("MBD-A020"));
    for (int k = 0; k < 4; ++k) {
        x = 0.5 * k;
        rec.sample(0.1 * k);
    }
    CHECK(rec.samples() == 4);
    CHECK(rec.column("x_squared")[3] == 2.25);
    CHECK_THROWS_WITH(rec.add("late", [] { return 0.0; }), ContainsSubstring("MBD-A021"));
    CHECK_THROWS_WITH(rec.column("y"), ContainsSubstring("MBD-A022"));

    const std::filesystem::path path = std::filesystem::temp_directory_path() / "mbd_recorder_test.csv";
    rec.write_csv(path.string());
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    CHECK(line == "time,x,x_squared");
    std::vector<std::string> rows;
    while (std::getline(in, line)) rows.push_back(line);
    in.close();
    std::filesystem::remove(path);
    REQUIRE(rows.size() == 4);
    CHECK(rows[3] == "0.30000000000000004,1.5,2.25");   // 17 digits: the values read back exactly

    CHECK_THROWS_WITH(rec.write_csv((std::filesystem::temp_directory_path() / "no_such_dir" / "x.csv").string()),
                      ContainsSubstring("MBD-A023"));
}
