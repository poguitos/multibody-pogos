// Task 7.11 (D72): mass properties. The chassis body from its own centre of
// mass and full inertia tensor and the masses it carries (a driver, fuel, a
// payload); the roll-up report of the built car. The plan's test: the report
// matches the inputs.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <sstream>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/vehicle/mass_properties.hpp"
#include "mbd/vehicle/suspension/mass.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using Catch::Matchers::ContainsSubstring;
using namespace mbd;

namespace {

/// A part of the car as the inputs describe it, in the chassis frame at the
/// design position: its mass, centre of mass and inertia about it.
struct Part {
    Real m;
    Vec3 c;
    Mat3 J;
};

/// The parallel-axis term, written out here: m (|d|^2 1 - d d^T).
Mat3 shifted(Real m, const Vec3& d) { return m * (d.dot(d) * Mat3::Identity() - d * d.transpose()); }

/// A uniform thin disc of mass m and radius R, its axis along Y (a wheel).
Mat3 disc(Real m, Real R) { return Mat3(Vec3(m * R * R / 4.0, m * R * R / 2.0, m * R * R / 4.0).asDiagonal()); }

/// A uniform rod of mass m and radius r from a to b, about its midpoint:
/// m (3 r^2 + L^2) / 12 across it, m r^2 / 2 along it.
Mat3 rod(Real m, const Vec3& a, const Vec3& b, Real r)
{
    const Vec3 u = (b - a).normalized();
    const Real L = (b - a).norm();
    return m * (3.0 * r * r + L * L) / 12.0 * (Mat3::Identity() - u * u.transpose())
         + 0.5 * m * r * r * u * u.transpose();
}

Real max_abs(const Mat3& A) { return A.cwiseAbs().maxCoeff(); }

} // namespace

// The plan's test. A car of simple front corners and trailing arms at the
// back, its axles unequally placed and tracked; the chassis with its centre of
// mass off the frame's origin and a full inertia tensor; a driver (with its
// own inertia), fuel and a payload. Every number the report gives is
// computed here from the inputs alone, part by part.
TEST_CASE("Mass properties: the roll-up report matches the inputs", "[mass]")
{
    VehicleTemplate t = VehicleTemplate::DefaultSedan();
    t.name = "mass_test";
    t.front_axle_x = 1.2;
    t.rear_axle_x = 1.5;
    t.rear_axle.suspension_type = SuspensionType::TrailingArm;
    t.rear_axle.half_track = 0.76;
    t.chassis.mass = 1250.0;
    t.chassis.com = Vec3(0.08, 0.012, 0.21);
    Mat3 J_chassis;
    J_chassis << 520.0, 3.0, -60.0,
                 3.0, 2100.0, -2.0,
                 -60.0, -2.0, 2300.0;
    t.chassis.inertia = J_chassis;
    t.chassis.lumped_masses = {
        {"driver", 75.0, Vec3(0.05, 0.37, 0.12), Mat3(Vec3(4.0, 4.5, 1.2).asDiagonal())},
        {"fuel", 45.0, Vec3(-1.05, 0.0, -0.05), Mat3::Zero()},
        {"payload", 60.0, Vec3(-1.75, -0.10, 0.05), Mat3::Zero()},
    };

    kernel::System sys;
    const VehicleHandle vh = build_vehicle(sys, t);
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    const MassReport r = mass_report(sim, vh);
    std::ostringstream text;
    text << r;
    INFO(text.str());

    // --- The parts, from the inputs. At the design position every wheel
    // centre is at the chassis frame's Z = 0 (the front ones at their axle's
    // X and half-track, the simple corners' sliders massless); each trailing
    // arm is a rod of arm_mass from its pivot to its wheel centre with its
    // hub, upright_mass, a sphere of lump_radius at the wheel centre
    // (mbd/vehicle/suspension/mass.hpp); the wheels uniform discs.
    const Real x_f = 1.2, x_r = -1.5, t_f = t.front_axle.half_track, t_r = 0.76;
    const Real R_f = t.front_axle.tyre_body().free_radius, R_r = t.rear_axle.tyre_body().free_radius;
    const Real m_wf = t.front_axle.wheel_mass, m_wr = t.rear_axle.wheel_mass;
    const Real m_arm = t.rear_axle.arm_mass, m_hub = t.rear_axle.upright_mass;
    std::vector<Part> parts{{1250.0, t.chassis.com, J_chassis}};
    for (const LumpedMass& l : t.chassis.lumped_masses) parts.push_back({l.mass, l.position, l.inertia});
    for (const Real y : {t_f, -t_f}) parts.push_back({m_wf, Vec3(x_f, y, 0.0), disc(m_wf, R_f)});
    for (const Real side : {1.0, -1.0}) {
        const Vec3 wc(x_r, side * t_r, 0.0);
        const Vec3 o = t.rear_axle.trailing_arm.pivot_offset;
        const Vec3 pivot = wc + Vec3(o.x(), side * o.y(), o.z());
        parts.push_back({m_wr, wc, disc(m_wr, R_r)});
        parts.push_back({m_arm, 0.5 * (pivot + wc), rod(m_arm, pivot, wc, rod_radius)});
        parts.push_back({m_hub, wc, 0.4 * m_hub * lump_radius * lump_radius * Mat3::Identity()});
    }

    Real M = 0.0;
    Vec3 first = Vec3::Zero();
    for (const Part& p : parts) {
        M += p.m;
        first += p.m * p.c;
    }
    const Vec3 cg = first / M;
    Mat3 J = Mat3::Zero();
    for (const Part& p : parts) J += p.J + shifted(p.m, p.c - cg);

    // Masses: 1250 + 75 + 45 + 60 + 4 x 40 + 2 x (5 + 15) = 1630 kg. Sums of
    // some twenty terms of up to 1630 kg: rounding below 1e-11 kg.
    CHECK(M == 1630.0);
    CHECK(std::abs(r.total_mass - M) < 1e-9);
    CHECK(std::abs(t.total_mass() - M) < 1e-9);
    // Positions through the bodies' poses, the chassis pitched: rounding of
    // order 1e-16 of the 2 m lengths.
    INFO("centre of mass: difference " << (r.centre_of_mass - cg).cwiseAbs().maxCoeff() << " m");
    CHECK((r.centre_of_mass - cg).cwiseAbs().maxCoeff() < 1e-12);
    // Inertias: twenty terms of up to M |r|^2 = 6500 kg m^2, rounding below
    // 20 x 6500 x 1.1e-16 = 1.4e-11 kg m^2.
    INFO("inertia: largest difference " << max_abs(r.inertia - J) << " kg m^2 of " << max_abs(J));
    CHECK(max_abs(r.inertia - J) < 1e-9);
    CHECK(r.yaw_inertia() == r.inertia(2, 2));

    // --- Unsprung masses. A simple corner: its wheel (its slider has no
    // mass). A trailing arm turns about its hinge: a point at a fraction f of
    // the way from the hinge to the wheel centre rises f times as much as the
    // wheel centre (velocities are linear in the distance from the axis), and
    // the contact, below the centre in the vertical wheel plane, rises as
    // the centre does. So the wheel and the hub count whole and the arm, its
    // centre of mass midway, half: 40 + 15 + 5 / 2 = 57.5 kg.
    const std::array<Real, 4> unsprung{{m_wf, m_wf, m_wr + m_hub + 0.5 * m_arm, m_wr + m_hub + 0.5 * m_arm}};
    for (std::size_t c = 0; c < 4; ++c) {
        INFO("corner " << c << ": unsprung, difference " << r.unsprung[c] - unsprung[c] << " kg");
        CHECK(std::abs(r.unsprung[c] - unsprung[c]) < 1e-9);
    }

    // --- Where the car stands. Each axle's load at rest per wheel, W_a, is
    // half its share of the weight by the lever rule between the wheel
    // centres; its tyres are at their loaded radius r_a = R_a - W_a / k_a; the
    // chassis is pitched by theta, sin(theta) = (r_r - r_f) / (x_f - x_r), its
    // origin at h = r_f + x_f sin(theta) (set_vehicle_equilibrium). A point
    // p of the chassis frame is then at world X = p_x cos(theta) + p_z
    // sin(theta), Z = h - p_x sin(theta) + p_z cos(theta).
    const Real m_front = M * (cg.x() - x_r) / (x_f - x_r);
    const Real r_f = R_f - 0.5 * g_accel * m_front / t.front_axle.tyre_body().vertical_stiffness;
    const Real r_r = R_r - 0.5 * g_accel * (M - m_front) / t.rear_axle.tyre_body().vertical_stiffness;
    const Real s = (r_r - r_f) / (x_f - x_r), co = std::sqrt(1.0 - s * s);
    const Real h = r_f + x_f * s;
    // The height of the centre of mass above the flat road: its world Z.
    const Real cg_height = h - cg.x() * s + cg.z() * co;
    INFO("height of the centre of mass: difference " << r.cg_height - cg_height << " m; pitch " << std::asin(s)
                                                       << " rad");
    CHECK(std::abs(r.cg_height - cg_height) < 1e-12);

    // The contacts: below each wheel centre, on the road, in the vertical
    // wheel plane; in the chassis frame, r_a along the pitched vertical
    // (sin(theta), 0, -cos(theta)) from the centre.
    const std::array<Vec3, 4> centre{{Vec3(x_f, t_f, 0.0), Vec3(x_f, -t_f, 0.0), Vec3(x_r, t_r, 0.0),
                                      Vec3(x_r, -t_r, 0.0)}};
    for (std::size_t c = 0; c < 4; ++c) {
        const Real ra = c < 2 ? r_f : r_r;
        const Vec3 expected = centre[c] + ra * Vec3(s, 0.0, -co);
        INFO("corner " << c << ": contact, difference " << (r.contact[c] - expected).cwiseAbs().maxCoeff() << " m");
        CHECK((r.contact[c] - expected).cwiseAbs().maxCoeff() < 1e-12);
    }

    // --- The sprung mass at each corner, by the lever rule along the road
    // (world X and Y here): the sprung weight acts at the car's first moment
    // less each unsprung mass's at its contact (at world X = x_a cos(theta),
    // Y = its side's half-track); its share between the axles, then side to
    // side.
    const Real m_s = M - (unsprung[0] + unsprung[1] + unsprung[2] + unsprung[3]);
    CHECK(std::abs(r.sprung_mass - m_s) < 1e-9);
    Real Sx = M * (cg.x() * co + cg.z() * s), Sy = M * cg.y();
    for (std::size_t c = 0; c < 4; ++c) {
        Sx -= unsprung[c] * centre[c].x() * co;
        Sy -= unsprung[c] * centre[c].y();
    }
    const Real X_f = x_f * co, X_r = x_r * co;
    const Real front = m_s * (Sx / m_s - X_r) / (X_f - X_r);
    const std::array<Real, 4> sprung{{front * (Sy / m_s + t_f) / (2.0 * t_f),
                                      front * (t_f - Sy / m_s) / (2.0 * t_f),
                                      (m_s - front) * (Sy / m_s + t_r) / (2.0 * t_r),
                                      (m_s - front) * (t_r - Sy / m_s) / (2.0 * t_r)}};
    Real total = 0.0, moment_x = 0.0, moment_y = 0.0;
    for (std::size_t c = 0; c < 4; ++c) {
        INFO("corner " << c << ": sprung, difference " << r.sprung[c] - sprung[c] << " kg");
        CHECK(std::abs(r.sprung[c] - sprung[c]) < 1e-9);
        total += r.corner_mass(static_cast<int>(c));
        moment_x += r.corner_mass(static_cast<int>(c)) * centre[c].x() * co;
        moment_y += r.corner_mass(static_cast<int>(c)) * centre[c].y();
    }
    // The corners' masses carry the whole car: their sum is its mass, and
    // their moments about both ground axes its own.
    INFO("corners: mass " << total - M << " kg, moments " << moment_x - M * (cg.x() * co + cg.z() * s) << ", "
                          << moment_y - M * cg.y() << " kg m");
    CHECK(std::abs(total - M) < 1e-9);
    CHECK(std::abs(moment_x - M * (cg.x() * co + cg.z() * s)) < 1e-9);
    CHECK(std::abs(moment_y - M * cg.y()) < 1e-9);
}

// A double wishbone moves its upright on two arms, so its unsprung mass has no
// simple closed form: here it is found again from positions instead of
// velocities. The corner on its rig is moved by +-delta and +-2 delta; the
// rise of the parts' weight over the rise of the contact (a point carried by
// the upright, at the loaded radius below the centre at the design position),
// by central differences with Richardson's extrapolation (error of order
// delta^4).
TEST_CASE("Mass properties: a double wishbone's unsprung mass from positions", "[mass]")
{
    VehicleTemplate t = VehicleTemplate::SportsCar();
    kernel::System car;
    const VehicleHandle vh = build_vehicle(car, t);

    kernel::System sys;
    const CornerRig rig = build_corner_rig(sys, t, 0);
    Kinematics k(sys);
    REQUIRE(k.solve(50, 1e-14));
    const TyreBody tyre = t.front_axle.tyre_body();
    const Real loaded = tyre.free_radius - vh.design_tyre_load[0] / tyre.vertical_stiffness;
    const BodyIndex upright = rig.corners[0].upright_body;
    // The contact as corner_motion places it: the wheel circle's lowest point,
    // in the wheel plane (the plane normal to the upright's Y), carried by
    // the upright.
    const RigidBodyState u0 = k.state(upright);
    const Vec3 axis = u0.q_WB * Vec3::UnitY();
    const Vec3 down = (-Vec3::UnitZ() + Vec3::UnitZ().dot(axis) * axis).normalized();
    const Vec3 contact_B = u0.q_WB.conjugate() * Vec3(loaded * down);

    // The parts' weight's first moment (sum of m z) and the contact's height
    // with the wheel centre at travel tau.
    auto heights = [&](Real tau) {
        k.t = tau;
        REQUIRE(k.solve(50, 1e-14));
        Real weight = 0.0;
        for (int b = 1; b < sys.model.nbodies(); ++b) {
            if (b == rig.chassis_body) continue;
            const RigidBodyInertia& I = sys.model.inertia[static_cast<std::size_t>(b)];
            const RigidBodyState st = k.state(b);
            weight += I.mass * (st.p_WB + st.q_WB * I.com_B).z();
        }
        const RigidBodyState us = k.state(upright);
        return std::array<Real, 2>{{weight, (us.p_WB + us.q_WB * contact_B).z()}};
    };
    const Real delta = 1e-3;
    auto ratio = [&](Real d) {
        const auto up = heights(d), down = heights(-d);
        return (up[0] - down[0]) / (up[1] - down[1]);
    };
    const Real d1 = ratio(delta), d2 = ratio(2.0 * delta);
    const Real extrapolated = (4.0 * d1 - d2) / 3.0;

    // The same corner's unsprung mass as the builder found it, from
    // velocities (corner_motion). The extrapolation's own error: delta^4 over
    // the linkage's length to the fourth (0.3 m), 1e-10 of the mass; the
    // solve's 1e-14 m over 2 mm, 5e-12. Allowed 1e-8 kg (2e-10 of 50 kg).
    INFO("unsprung: builder " << vh.corners[0].unsprung_mass << " kg; from positions, difference "
                              << vh.corners[0].unsprung_mass - extrapolated << " (delta alone "
                              << vh.corners[0].unsprung_mass - d1 << ", 2 delta " << vh.corners[0].unsprung_mass - d2
                              << "); a whole wheel and upright and half the arms: "
                              << t.front_axle.wheel_mass + t.front_axle.upright_mass + t.front_axle.arm_mass);
    CHECK(std::abs(vh.corners[0].unsprung_mass - extrapolated) < 1e-8);
    CHECK(vh.corners[1].unsprung_mass == vh.corners[0].unsprung_mass);

    // The whole car: its corners carry its mass, every body is counted.
    kernel::Simulator sim(car);
    set_vehicle_equilibrium(sim, vh);
    const MassReport r = mass_report(sim, vh);
    std::ostringstream text;
    text << r;
    INFO(text.str());
    INFO("design loads per wheel over g: front " << vh.design_tyre_load[0] / g_accel << ", rear "
                                                 << vh.design_tyre_load[1] / g_accel << " kg");
    CHECK(std::abs(r.total_mass - t.total_mass()) < 1e-9);
    Real total = 0.0;
    for (int c = 0; c < 4; ++c) total += r.corner_mass(c);
    CHECK(std::abs(total - r.total_mass) < 1e-9);
}

TEST_CASE("Mass properties: the chassis body", "[mass]")
{
    // Without a centre of mass, inertia or lumped masses, the chassis is the
    // solid box it was before task 7.11, bitwise.
    const ChassisConfig plain;
    const RigidBodyInertia a = chassis_inertia(plain);
    const RigidBodyInertia b = RigidBodyInertia::from_solid_box(plain.mass, plain.half_extents);
    CHECK(a.mass == b.mass);
    CHECK(a.com_B == b.com_B);
    CHECK(a.I_com_B == b.I_com_B);

    // A point mass on the chassis: the combined centre of mass and inertia by
    // hand. 1400 kg at the origin (the box) and 100 kg at (2, 0, 0): the
    // centre of mass at 200 / 1500 = 0.1333 m; about Y, the box's
    // 1400 (1.5^2 + 0.3^2) / 3 = 1092 kg m^2 plus 1400 x 0.1333^2 and
    // 100 x 1.8667^2 (the masses' distances from the new centre).
    ChassisConfig loaded = plain;
    loaded.lumped_masses = {{"ballast", 100.0, Vec3(2.0, 0.0, 0.0), Mat3::Zero()}};
    const RigidBodyInertia c = chassis_inertia(loaded);
    const Real x = 200.0 / 1500.0;
    const Real Iyy = 1400.0 * (1.5 * 1.5 + 0.3 * 0.3) / 3.0 + 1400.0 * x * x + 100.0 * (2.0 - x) * (2.0 - x);
    CHECK(c.mass == 1500.0);
    CHECK(std::abs(c.com_B.x() - x) < 1e-15);
    CHECK(std::abs(c.I_com_B(1, 1) - Iyy) < 1e-10);

    // Refused: a tensor that is not symmetric, one with a negative principal
    // moment, one whose moments break the triangle inequality, a negative
    // lumped mass, a chassis without mass, a centre of mass that is not a
    // number.
    auto refused = [](const ChassisConfig& cfg, const char* why) {
        INFO(why);
        CHECK_THROWS_WITH(chassis_inertia(cfg), ContainsSubstring("MBD-V008"));
    };
    ChassisConfig bad = plain;
    Mat3 J = Mat3(Vec3(500.0, 2000.0, 2200.0).asDiagonal());
    J(0, 2) = 10.0;
    bad.inertia = J;
    refused(bad, "not symmetric");
    bad.inertia = Mat3(Vec3(-1.0, 2000.0, 2200.0).asDiagonal());
    refused(bad, "a negative moment");
    bad.inertia = Mat3(Vec3(100.0, 200.0, 2200.0).asDiagonal());
    refused(bad, "the triangle inequality");
    bad = plain;
    bad.lumped_masses = {{"negative", -1.0, Vec3::Zero(), Mat3::Zero()}};
    refused(bad, "a negative lumped mass");
    bad.lumped_masses = {{"driver", 75.0, Vec3::Zero(), Mat3(Vec3(1.0, 1.0, 3.0).asDiagonal())}};
    refused(bad, "a lumped mass's triangle inequality");
    bad = plain;
    bad.mass = 0.0;
    refused(bad, "no mass");
    bad = plain;
    bad.com = Vec3(std::nan(""), 0.0, 0.0);
    refused(bad, "a centre of mass that is not a number");

    // And through the builder.
    VehicleTemplate t = VehicleTemplate::DefaultSedan();
    t.chassis.inertia = Mat3(Vec3(100.0, 200.0, 2200.0).asDiagonal());
    kernel::System sys;
    CHECK_THROWS_WITH(build_vehicle(sys, t), ContainsSubstring("MBD-V008"));
}
