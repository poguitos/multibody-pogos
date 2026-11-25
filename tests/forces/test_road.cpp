// The road (plan task 7.2, D61): each road's height and normal against its
// closed form, every normal against central differences of the heights, the
// gridded road on planes and nodes, the ISO 8608 roads against their classes'
// spectra, and the tyres on roads.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "mbd/forces/road.hpp"
#include "mbd/forces/tire.hpp"

using namespace mbd;
using Catch::Matchers::WithinAbs;

namespace {

/// The normal a road should have at (x, y), from central differences of its
/// heights: (-dh/dx, -dh/dy, 1), normalised. With a step of 1e-6 m the
/// truncation error is about 1e-12 times the third derivative and the
/// rounding about 1e-16 |h| / 1e-6; the roads below keep both under 1e-9.
Vec3 normal_by_differences(const Road& road, Real x, Real y)
{
    const Real h = 1e-6;
    const Real dx = (road.at(x + h, y).height - road.at(x - h, y).height) / (2.0 * h);
    const Real dy = (road.at(x, y + h).height - road.at(x, y - h).height) / (2.0 * h);
    return Vec3(-dx, -dy, 1.0).normalized();
}

void check_normals(const Road& road, const std::vector<Vec2>& points)
{
    for (const Vec2& p : points) {
        const RoadPoint r = road.at(p.x(), p.y());
        INFO(road.name() << " at (" << p.x() << ", " << p.y() << "): normal " << r.normal.transpose()
                         << ", from differences " << normal_by_differences(road, p.x(), p.y()).transpose());
        CHECK((r.normal - normal_by_differences(road, p.x(), p.y())).norm() < 1e-8);
        CHECK(std::abs(r.normal.norm() - 1.0) < 1e-15);
    }
}

} // namespace

TEST_CASE("Road: a flat road is level at its height everywhere", "[road]")
{
    const FlatRoad road(0.25, 0.8);
    for (const Vec2& p : {Vec2(0.0, 0.0), Vec2(-30.0, 12.0), Vec2(1e4, -1e4)}) {
        const RoadPoint r = road.at(p.x(), p.y());
        CHECK(r.height == 0.25);
        CHECK(r.normal == Vec3::UnitZ());
        CHECK(r.friction == 0.8);
    }
}

TEST_CASE("Road: ramp, bump, kerb and sine against their closed forms", "[road]")
{
    // Ramp: 0.1 m over 2 m from x = 5. Halfway, h = 0.05 and the slope 0.05,
    // the normal (-0.05, 0, 1) / sqrt(1.0025).
    const RampRoad ramp(5.0, 2.0, 0.1);
    CHECK(ramp.at(4.0, 0.0).height == 0.0);
    CHECK_THAT(ramp.at(6.0, 3.0).height, WithinAbs(0.05, 1e-15));
    CHECK(ramp.at(8.0, 0.0).height == 0.1);
    CHECK((ramp.at(6.0, 0.0).normal - Vec3(-0.05, 0.0, 1.0) / std::sqrt(1.0025)).norm() < 1e-15);

    // Bump: 0.04 m, 0.5 m long from x = 1. At its middle h = 0.04, level; a
    // quarter in, h = 0.02 and the slope 0.04 (2 pi / 0.5) / 2 = 0.08 pi.
    const BumpRoad bump(1.0, 0.5, 0.04);
    CHECK_THAT(bump.at(1.25, 0.0).height, WithinAbs(0.04, 1e-15));
    CHECK((bump.at(1.25, 0.0).normal - Vec3::UnitZ()).norm() < 1e-15);
    CHECK_THAT(bump.at(1.125, 0.0).height, WithinAbs(0.02, 1e-15));
    const Real s = 0.08 * pi;
    CHECK((bump.at(1.125, 0.0).normal - Vec3(-s, 0.0, 1.0) / std::sqrt(1.0 + s * s)).norm() < 1e-14);
    CHECK(bump.at(0.9, 0.0).height == 0.0);
    CHECK(bump.at(1.6, 0.0).height == 0.0);

    // Kerb: from y = 3 to 3.5, 0.06 m high, 0.1 m chamfers. Halfway up the
    // inner chamfer h = 0.03 and dh/dy = 0.6, the normal (0, -0.6, 1) / sqrt(1.36);
    // on the top 0.06, level; beyond, 0.
    const KerbRoad kerb(3.0, 3.5, 0.06, 0.1);
    CHECK_THAT(kerb.at(10.0, 3.05).height, WithinAbs(0.03, 1e-15));
    CHECK((kerb.at(10.0, 3.05).normal - Vec3(0.0, -0.6, 1.0) / std::sqrt(1.36)).norm() < 1e-15);
    CHECK_THAT(kerb.at(-4.0, 3.45).height, WithinAbs(0.03, 1e-15));
    CHECK((kerb.at(-4.0, 3.45).normal - Vec3(0.0, 0.6, 1.0) / std::sqrt(1.36)).norm() < 1e-15);
    CHECK(kerb.at(0.0, 3.25).height == 0.06);
    CHECK(kerb.at(0.0, 2.9).height == 0.0);
    CHECK(kerb.at(0.0, 3.6).height == 0.0);

    // Sine: 0.01 m, 2 m. At x = 0.5 (a quarter wave) the crest, level; at 0
    // h = 0 and the slope 0.01 pi.
    const SineRoad sine(0.01, 2.0);
    CHECK_THAT(sine.at(0.5, 0.0).height, WithinAbs(0.01, 1e-17));
    CHECK((sine.at(0.5, 0.0).normal - Vec3::UnitZ()).norm() < 1e-15);
    CHECK_THAT(sine.at(0.0, 0.0).height, WithinAbs(0.0, 1e-17));
    CHECK((sine.at(0.0, 0.0).normal - Vec3(-0.01 * pi, 0.0, 1.0) / std::sqrt(1.0 + 1e-4 * pi * pi)).norm() < 1e-15);

    // Every normal is that of the heights, away from the ramp's and the
    // kerb's corners, where the slope jumps.
    check_normals(ramp, {Vec2(5.5, 0.0), Vec2(6.7, -2.0), Vec2(3.0, 0.0), Vec2(9.0, 1.0)});
    check_normals(bump, {Vec2(1.1, 0.0), Vec2(1.31, 4.0), Vec2(1.45, 0.0)});
    check_normals(kerb, {Vec2(0.0, 3.02), Vec2(2.0, 3.3), Vec2(-1.0, 3.47)});
    check_normals(sine, {Vec2(0.13, 0.0), Vec2(0.77, 1.0), Vec2(-3.4, 0.0)});
}

TEST_CASE("Road: a gridded road reproduces a plane, passes through its nodes, and continues level",
          "[road][grid]")
{
    // Bilinear interpolation is exact for h = a + b x + c y, so a grid of such
    // heights is that plane everywhere inside it, normal (-b, -c, 1) normalised.
    const Real a = 0.1, b = 0.02, c = -0.03;
    const int nx = 7, ny = 5;
    std::vector<Real> plane;
    std::vector<Real> wavy;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Real x = -1.0 + 0.5 * i, y = 2.0 + 0.25 * j;
            plane.push_back(a + b * x + c * y);
            wavy.push_back(0.01 * std::sin(1.3 * i + 0.7 * j * j));
        }
    }
    const GridRoad flat_plane(-1.0, 2.0, 0.5, 0.25, nx, ny, plane);
    for (const Vec2& p : {Vec2(-0.83, 2.11), Vec2(1.0, 3.0), Vec2(1.97, 2.99)}) {
        const RoadPoint r = flat_plane.at(p.x(), p.y());
        CHECK_THAT(r.height, WithinAbs(a + b * p.x() + c * p.y(), 1e-15));
        CHECK((r.normal - surface_normal(b, c)).norm() < 1e-15);
    }
    const GridRoad grid(-1.0, 2.0, 0.5, 0.25, nx, ny, wavy);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            CHECK_THAT(grid.at(-1.0 + 0.5 * i, 2.0 + 0.25 * j).height,
                       WithinAbs(wavy[static_cast<std::size_t>(i + nx * j)], 1e-15));
        }
    }
    // Inside a cell the normal is that of the bilinear heights.
    check_normals(grid, {Vec2(-0.7, 2.1), Vec2(0.6, 2.6), Vec2(1.8, 2.9)});
    // Beyond a corner of the grid: that corner's height, level.
    CHECK(grid.at(5.0, 10.0).height == wavy.back());
    CHECK(grid.at(5.0, 10.0).normal == Vec3::UnitZ());
}

TEST_CASE("Road: parameters out of range are refused (MBD-F050)", "[road]")
{
    const auto code = [](auto make) {
        try {
            make();
        } catch (const MbdError& e) {
            return std::string(e.what()).substr(0, 8);
        }
        return std::string("none");
    };
    CHECK(code([] { RampRoad(0.0, 0.0, 0.1); }) == "MBD-F050");
    CHECK(code([] { BumpRoad(0.0, -1.0, 0.1); }) == "MBD-F050");
    CHECK(code([] { KerbRoad(1.0, 1.1, 0.05, 0.1); }) == "MBD-F050");   // chamfers wider than the kerb
    CHECK(code([] { SineRoad(0.01, 0.0); }) == "MBD-F050");
    CHECK(code([] { GridRoad(0.0, 0.0, 1.0, 1.0, 2, 2, {0.0, 0.0, 0.0}); }) == "MBD-F050");
    CHECK(code([] { FlatRoad(0.0, -0.5); }) == "MBD-F050");
    Iso8608Params p;
    p.n_max = p.n_min;
    CHECK(code([&] { Iso8608Road{p}; }) == "MBD-F050");
}

// ============================================================================
// ISO 8608
// ============================================================================

TEST_CASE("Road: ISO 8608's spectra at their reference frequency, falling as n^-2", "[road][iso8608]")
{
    // ISO 8608, table: the geometric means of the classes' bands at
    // n0 = 0.1 cycles/m, in 1e-6 m^3: A 16, B 64, C 256, D 1024, E 4096,
    // F 16384, G 65536, H 262144. Waviness w = 2.
    const Real table[] = {16, 64, 256, 1024, 4096, 16384, 65536, 262144};
    for (int k = 0; k < 8; ++k) {
        const auto c = static_cast<Iso8608Class>(k);
        CHECK_THAT(iso8608_psd_at_n0(c), WithinAbs(table[k] * 1e-6, 1e-18 * table[k]));
        CHECK_THAT(iso8608_psd(c, 0.1), WithinAbs(table[k] * 1e-6, 1e-18 * table[k]));
        CHECK_THAT(iso8608_psd(c, 0.2), WithinAbs(table[k] * 1e-6 / 4.0, 1e-18 * table[k]));
        CHECK_THAT(iso8608_psd(c, 0.01), WithinAbs(table[k] * 1e-6 * 100.0, 1e-16 * table[k]));
    }
}

TEST_CASE("Road: a generated ISO 8608 road reproduces its class's spectrum, every class", "[road][iso8608]")
{
    // A road of length L is periodic with period L, its samples j L / N,
    // j < N, taken through the road's own interface (at(x, 0); between
    // samples the interpolation passes through them). The one-sided
    // periodogram of those N samples at n_k = k / L is
    //
    //     G(n_k) = |Z_k|^2 / (2 / L),  Z_k = (2 / N) sum_j h_j e^(-2 pi i k j / N),
    //
    // and for the road's components, each a cosine of amplitude A_k at n_k,
    // |Z_k| = A_k. The class's spectrum G_d(n_k) = G_d(n0) (n_k / n0)^-2 is
    // reproduced if every in-band frequency holds G_d(n_k) and every other
    // nothing. The DFT here is the definition, independent of the road's code;
    // its rounding is about 1e-16 of the profile's RMS per term, which leaves
    // the smallest in-band G (at n_max) correct to about 1e-12 relative:
    // allowed 1e-9. Out of the band, allowed 1e-12 of the smallest in-band G.
    const Real table[] = {16, 64, 256, 1024, 4096, 16384, 65536, 262144};
    for (int c = 0; c < 8; ++c) {
        Iso8608Params p;
        p.road_class = static_cast<Iso8608Class>(c);
        p.seed = 1000 + static_cast<std::uint64_t>(c);
        p.length = 200.0;
        const Iso8608Road road(p);
        const Real L = p.length;
        const int N = static_cast<int>(road.samples().size());
        REQUIRE(N == static_cast<int>(std::ceil(10 * 2.83 * L)));   // 10 samples per shortest wavelength
        std::vector<Real> h(static_cast<std::size_t>(N));
        for (int j = 0; j < N; ++j) h[static_cast<std::size_t>(j)] = road.at(j * L / N, 0.0).height;

        const Real G0 = table[c] * 1e-6;
        const Real smallest = G0 * (0.1 / 2.83) * (0.1 / 2.83);
        std::vector<std::complex<Real>> twiddle(static_cast<std::size_t>(N));   // e^(-2 pi i m / N)
        for (int m = 0; m < N; ++m) twiddle[static_cast<std::size_t>(m)] = std::polar(Real(1.0), -2.0 * pi * m / N);
        Real worst_in = 0.0, worst_out = 0.0, at_n0 = 0.0;
        for (int k = 1; k <= static_cast<int>(3.5 * L); ++k) {
            std::complex<Real> Z = 0.0;
            for (int j = 0; j < N; ++j) {
                const auto m = static_cast<std::size_t>(static_cast<long long>(k) * j % N);   // exact angle index
                Z += h[static_cast<std::size_t>(j)] * twiddle[m];
            }
            Z *= 2.0 / N;
            const Real G = std::norm(Z) / (2.0 / L);
            const Real n = k / L;
            if (n >= 0.011 && n <= 2.83) {
                const Real expected = G0 * (0.1 / n) * (0.1 / n);
                worst_in = std::max(worst_in, std::abs(G - expected) / expected);
                if (k == 20) at_n0 = G;   // n = 0.1
            } else {
                worst_out = std::max(worst_out, G / smallest);
            }
        }
        INFO("class " << char('A' + c) << ": G(n0) " << at_n0 << " m^3 against " << G0
                      << "; worst in the band " << worst_in << " relative; out of it " << worst_out
                      << " of the smallest in-band G");
        CHECK(worst_in < 1e-9);
        CHECK(worst_out < 1e-12);
        CHECK(std::abs(at_n0 - G0) < 1e-9 * G0);
    }
}

TEST_CASE("Road: an ISO 8608 road is fixed by its seed, smooth, and the same across the road",
          "[road][iso8608]")
{
    Iso8608Params p;
    p.road_class = Iso8608Class::C;
    p.length = 100.0;
    p.seed = 42;
    const Iso8608Road a(p), b(p);
    p.seed = 43;
    const Iso8608Road other(p);
    CHECK(a.samples() == b.samples());
    CHECK(a.samples() != other.samples());

    // The same across the road, and one period L later.
    CHECK(a.at(12.34, 0.0).height == a.at(12.34, -3.0).height);
    CHECK_THAT(a.at(12.34 + p.length, 0.0).height, WithinAbs(a.at(12.34, 0.0).height, 1e-12));

    // Its slope is continuous: at a sample, the slopes from either side agree
    // to the step's first-order change (Catmull-Rom is C1), and between samples
    // the normal is that of the heights.
    const Real x_j = 37 * a.spacing();
    const Real eps = 1e-9 * a.spacing();
    CHECK((a.at(x_j - eps, 0.0).normal - a.at(x_j + eps, 0.0).normal).norm() < 1e-6);
    check_normals(a, {Vec2(1.234, 0.0), Vec2(55.5, 2.0), Vec2(99.97, 0.0)});
}

// ============================================================================
// The tyres on roads
// ============================================================================

TEST_CASE("Road: a tyre on a bump's flank pushes along the road's normal", "[road][tyre]")
{
    // A wheel at rest, centre at x = 1.125 and z = 0.365, a quarter into a
    // bump of 0.04 m and 0.5 m (h = H (1 - cos(2 pi (x - 1) / L)) / 2, its
    // slope H pi / L sin(...)). The contact from the wheel plane (task 7.3):
    // the road's normal under the centre, n_c = (-s, 0, 1) / sqrt(1 + s^2)
    // with s = 0.08 pi, gives the lowest point P = c - R n_c (the wheel's
    // plane holds n_c); the road at P's x gives the height and the normal
    // n_P there, the depth (h(P) - P_z) n_P,z = 0.011921 m; with no damping
    // and no slip (the wheel at rest) the tyre pushes k depth along n_P,
    // computed here from the bump's closed form. (The point below the centre,
    // as the contact was before, gave 0.004849 m; the exact depth of the
    // wheel's circle in this road, R less the distance from the centre to the
    // road's nearest point, is 0.013302 m: the journal's data.)
    const Real H = 0.04, L = 0.5, R = 0.35;
    auto h = [&](Real x) { return H * (1.0 - std::cos(2.0 * pi * (x - 1.0) / L)) / 2.0; };
    auto normal = [&](Real x) -> Vec3 {   // a Vec3, not an Eigen expression of a temporary
        const Real s = H * pi / L * std::sin(2.0 * pi * (x - 1.0) / L);
        return Vec3(-s, 0.0, 1.0) / std::sqrt(1.0 + s * s);
    };
    const Vec3 c(1.125, 0.0, 0.35 + 0.02 - 0.005);
    const Vec3 P = c - R * normal(c.x());
    const Vec3 n_P = normal(P.x());
    const Real depth = (h(P.x()) - P.z()) * n_P.z();
    const auto bump = std::make_shared<BumpRoad>(1.0, L, H);
    FullTireForce tyre(1, R, 200000.0, 0.0, PacejkaTireParams::DefaultPassengerCar(), bump);
    std::vector<RigidBodyState> states(2);
    states[1] = RigidBodyState(c, Quat::Identity(), Vec3::Zero(), Vec3::Zero());
    std::vector<RigidBodyForces> forces(2);
    tyre.apply(states, forces);
    const Vec3 expected = 200000.0 * depth * n_P;
    INFO("force " << forces[1].f_W.transpose() << ", expected " << expected.transpose() << "; depth " << depth);
    CHECK_THAT(depth, WithinAbs(0.011921, 1e-6));
    CHECK((forces[1].f_W - expected).norm() < 1e-9 * expected.norm());

    // The vertical tyre, on a ramp's slope (0.05, a plane between x = 0 and
    // 2 m): the lowest point is along the normal, still on the plane, so the
    // depth is exact: R less the centre's distance from the plane,
    // R - (z_c - h(x_c)) n_z = 0.35 - 0.346 n_z.
    const auto ramp = std::make_shared<RampRoad>(0.0, 2.0, 0.1);
    TireContactForce vertical(1, R, 200000.0, 0.0, ramp);
    states[1] = RigidBodyState(Vec3(1.0, 0.0, 0.35 + 0.05 - 0.004), Quat::Identity(), Vec3::Zero(),
                               Vec3::Zero());
    forces.assign(2, RigidBodyForces{});
    vertical.apply(states, forces);
    const Vec3 m = ramp->at(1.0, 0.0).normal;
    const Real ramp_depth = R - 0.346 * m.z();
    CHECK((forces[1].f_W - 200000.0 * ramp_depth * m).norm() < 1e-9 * 800.0);
    CHECK_THAT(vertical.get_deflection(states), WithinAbs(ramp_depth, 1e-15));
}

TEST_CASE("Road: the road's friction scales a tyre's peak force and not its stiffness", "[road][tyre]")
{
    // A wheel loaded to Fz = k d = 200000 * 0.02 = 4000 N (the tyre's nominal
    // load, where its lateral friction is mu0 = 1.0), sliding sideways at
    // speeds that sweep its slip angle past the peak. The Magic Formula's
    // peak is D = mu Fz times the road's friction, reached where
    // C atan(...) = pi / 2 (C = 1.3 > 1). On a road of friction 0.5 the peak
    // is 2000 N, on 1.0 4000 N. Sampled every 3e-4 rad, a maximum falls short
    // of the peak by at most |f''| (1.5e-4)^2 / 2, f'' = -D theta'^2 with
    // theta = C atan(u) the angle under the sine: theta' is about 6 rad^-1
    // on the road of 0.5 (B = K / (C D) = 17.3), so the shortfall is about
    // 4e-7 relative: allowed 1e-5. Near zero slip both give the cornering
    // stiffness K = BCD, the same on both roads: at 1e-7 rad they differ by
    // the cubic term, about (B alpha)^2 = 3e-12 relative: allowed 1e-9.
    auto peak_and_slope = [](Real friction) {
        FullTireForce tyre(1, 0.35, 200000.0, 0.0, PacejkaTireParams::DefaultPassengerCar(),
                           std::make_shared<FlatRoad>(0.0, friction));
        std::vector<RigidBodyState> states(2);
        std::vector<RigidBodyForces> forces(2);
        auto lateral_force = [&](Real alpha) {
            // Forward at 20 m/s, sideways at -20 tan(alpha): slip angle alpha.
            // The wheel spins at free rolling, 20 / R_eff with R_eff = 0.35 -
            // 0.02 / 2 (task 7.3: the tyre takes the spin from the wheel).
            states[1] = RigidBodyState(Vec3(0.0, 0.0, 0.33), Quat::Identity(),
                                       Vec3(20.0, -20.0 * std::tan(alpha), 0.0), Vec3(0.0, 20.0 / 0.34, 0.0));
            forces.assign(2, RigidBodyForces{});
            tyre.apply(states, forces);
            return tyre.get_Fy();
        };
        Real peak = 0.0;
        for (int i = 1; i <= 4000; ++i) peak = std::max(peak, lateral_force(i * 3e-4));
        return std::pair<Real, Real>(peak, lateral_force(1e-7) / 1e-7);
    };
    const auto [peak_dry, slope_dry] = peak_and_slope(1.0);
    const auto [peak_wet, slope_wet] = peak_and_slope(0.5);
    INFO("peaks " << peak_dry << " and " << peak_wet << " N; slopes " << slope_dry << " and " << slope_wet);
    CHECK(std::abs(peak_dry - 4000.0) < 1e-5 * 4000.0);
    CHECK(std::abs(peak_wet - 2000.0) < 1e-5 * 2000.0);
    CHECK(std::abs(slope_wet - slope_dry) < 1e-9 * slope_dry);
}

TEST_CASE("Road: a tyre with no road is refused (MBD-F051)", "[road][tyre]")
{
    FullTireForce tyre(1, 0.35, 200000.0, 0.0, PacejkaTireParams::DefaultPassengerCar());
    REQUIRE(tyre.road != nullptr);   // a level road by default
    CHECK(tyre.road->at(3.0, 4.0).height == 0.0);
    tyre.road = nullptr;
    std::vector<RigidBodyState> states(2);
    std::vector<RigidBodyForces> forces(2);
    try {
        tyre.apply(states, forces);
        FAIL("no error");
    } catch (const MbdError& e) {
        CHECK(std::string(e.what()).rfind("MBD-F051", 0) == 0);
    }
}
