// The Magic Formula tyre (plan task 7.4): the .tir reader, and the identities
// that the equations (Pacejka, 3rd ed., section 4.3.2, and the appendix of
// Besselink, Schmeitz and Pacejka, 2010) give at the nominal state of the
// book's Appendix 3 tyre (205/60R15, MF 6.1): load Fz = FNOMIN (dfz = 0),
// pressure NOMPRES (dpi = 0), no inclination. The comparison with an
// independent implementation is in test_magic_formula_mfeval.cpp.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <string>

#include "mbd/forces/magic_formula.hpp"
#include "mbd/forces/pacejka.hpp"
#include "mbd/io/tir_file.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace
{
    /// A file of tests/forces/data/.
    std::string data(const std::string& file)
    {
        return (std::filesystem::path(__FILE__).parent_path() / "data" / file).string();   // tests/forces/<this file>
    }

    MagicFormulaTyre appendix3() { return MagicFormulaTyre(io::read_tir(data("pacejka_a3_205_60R15.tir")).parameters); }

    MagicFormulaInput nominal(Real kappa, Real alpha)
    {
        MagicFormulaInput in;
        in.Fz = 4000.0;
        in.kappa = kappa;
        in.alpha = alpha;
        in.Vcx = 16.7;
        return in;
    }
}

TEST_CASE("Magic Formula: the reader takes the book's Appendix 3 tyre", "[tyre][mf][tir]")
{
    const io::TirFile f = io::read_tir(data("pacejka_a3_205_60R15.tir"));
    const MagicFormulaParameters& p = f.parameters;
    CHECK(p.version == MagicFormulaVersion::MF61);
    CHECK(p.FITTYP == 61);
    CHECK(p.FNOMIN == 4000.0);
    CHECK(p.UNLOADED_RADIUS == 0.3135);
    CHECK(p.PKY1 == -15.324);
    CHECK(p.PVX1 == 2.0283e-5);
    CHECK(p.RVY2 == 4.853e-4);
    CHECK(p.SSZ2 == 0.03869);
    CHECK(p.LFZO == 1.0);    // not in the file: scaling factors default to 1
    CHECK(p.LMUV == 0.0);    // except the slip-speed friction decay
    CHECK(p.PTX1 == 0.0);    // coefficients default to 0 (an MF 5.2 one here)
    // The header's keys are not the tyre's: listed, not refused.
    bool header = false;
    for (const auto& k : f.ignored) header = header || k == "MDI_HEADER.FILE_TYPE";
    CHECK(header);
}

TEST_CASE("Magic Formula: the reader refuses what it cannot read", "[tyre][mf][tir]")
{
    const std::string base = "[MODEL]\nFITTYP = 61\n[DIMENSION]\nUNLOADED_RADIUS = 0.3\n[VERTICAL]\nFNOMIN = 4000\n";
    CHECK_NOTHROW(io::parse_tir(base));
    CHECK_THROWS_WITH(io::parse_tir("[MODEL]\nFITTYP = 5\n"), ContainsSubstring("MBD-I022"));
    CHECK_THROWS_WITH(io::parse_tir("[UNITS]\nLENGTH = 'mm'\n" + base), ContainsSubstring("MBD-I021"));
    CHECK_THROWS_WITH(io::parse_tir("[MODEL]\nFITTYP = 61\n[VERTICAL]\nFNOMIN = 4000\n"),
                      ContainsSubstring("MBD-I023"));
    CHECK_THROWS_WITH(io::parse_tir(base + "PKY1 = abc\n"), ContainsSubstring("MBD-I024"));
    CHECK_THROWS_WITH(io::read_tir(data("no such file.tir")), ContainsSubstring("MBD-I020"));
    // Adams's name for MF 5.2, in a file without FITTYP; comments after '$'.
    const io::TirFile adams = io::parse_tir(
        "[MODEL]\nPROPERTY_FILE_FORMAT = 'PAC2002' $ the formulas\n[DIMENSION]\nUNLOADED_RADIUS = 0.3\n"
        "[VERTICAL]\nFNOMIN = 4000 $ nominal\n");
    CHECK(adams.parameters.version == MagicFormulaVersion::MF52);
    CHECK(adams.parameters.FNOMIN == 4000.0);
    // Parameters the tyre cannot evaluate are refused by the tyre.
    MagicFormulaParameters bad = io::parse_tir(base).parameters;
    bad.FNOMIN = 0.0;
    CHECK_THROWS_WITH(MagicFormulaTyre(bad), ContainsSubstring("MBD-F053"));
}

TEST_CASE("Magic Formula: the slip stiffnesses are the curves' slopes, at the book's nominal state",
          "[tyre][mf]")
{
    // At dfz = dpi = 0 and no inclination, (4.E15) gives Kxk = Fz pKx1 =
    // 4000 x 21.687 = 86748 N and (4.E25) Kya = pKy1 Fz sin(pKy4 atan(1 /
    // pKy2)) = -15.324 x 4000 sin(2.0005 atan(1 / 1.715)). The sine form's
    // slope at its centre is B C D, which (4.E16) and (4.E26) make the
    // stiffness, but for the epsilon in their denominators (1e-6 against C D
    // of 6600 N and 4700 N: 2e-10 relative). Central differences of 1e-7 in
    // the slip about each curve's centre (kappa = -SHx, alpha = -SHy)
    // have an error of the third derivative's order, 1e-14 of the slope, and
    // a rounding of 1e-16 x 4000 / 1e-7 = 4e-6 N: allowed 1e-8 relative.
    // The slip angle enters the formulas as it is (D63): the slope is in it.
    const MagicFormulaTyre tyre = appendix3();
    const MagicFormulaParameters& p = tyre.parameters();
    const Real Kxk = 4000.0 * p.PKX1;
    const Real Kya = p.PKY1 * 4000.0 * std::sin(p.PKY4 * std::atan(1.0 / p.PKY2));
    const MagicFormulaOutput at0 = tyre.evaluate(nominal(0.0, 0.0));
    CHECK_THAT(at0.Kxk, WithinRel(Kxk, 1e-15));
    CHECK_THAT(at0.Kya, WithinRel(Kya, 1e-15));

    const Real h = 1e-7;
    const Real SHx = p.PHX1;   // (4.E17) at dfz = 0
    const Real slope_x = (tyre.evaluate(nominal(-SHx + h, 0.0)).Fx0 - tyre.evaluate(nominal(-SHx - h, 0.0)).Fx0) / (2 * h);
    INFO("dFx0/dkappa " << slope_x << " N against Kxk " << Kxk);
    CHECK_THAT(slope_x, WithinRel(Kxk, 1e-8));

    // The lateral curve's centre: alpha_y = 0 at alpha = -SHy, SHy from the
    // shifts at zero inclination, (pHy1 + 0) + (0 - 0) / Kya = pHy1.
    const Real a0 = -p.PHY1;
    const Real slope_y = (tyre.evaluate(nominal(0.0, a0 + h)).Fy0 - tyre.evaluate(nominal(0.0, a0 - h)).Fy0) / (2 * h);
    INFO("dFy0/dalpha " << slope_y << " N/rad against Kya " << Kya);
    CHECK_THAT(slope_y, WithinRel(Kya, 1e-8));
}

TEST_CASE("Magic Formula: the peaks are mu Fz, shifted by SV, at the book's nominal state", "[tyre][mf]")
{
    // With C > 1 the sine form reaches D = mu Fz: mux = pDx1 = 1.0422 (4.E13),
    // muy = pDy1 = 0.8785 (4.E23) at dfz = dpi = 0; the curves are then lifted
    // by SVx = Fz pVx1 lambda'mux (4.E18) and SVy = Fz pVy1 lambda'muy
    // (4.E29), lambda' = 1 with every scaling factor 1. Swept every 2e-5 in
    // kappa and in alpha, the largest value falls short of the peak by at
    // most D (theta' dx)^2 / 8 for the angle theta under the sine: theta' is
    // at most B C (21 and 17 per unit slip here), so 6e-6 of D: allowed 1e-5.
    const MagicFormulaTyre tyre = appendix3();
    const MagicFormulaParameters& p = tyre.parameters();
    Real peak_x = -1e9, peak_y = 1e9;
    for (int i = 0; i <= 25000; ++i) {
        const Real s = i * 2e-5;
        peak_x = std::max(peak_x, tyre.evaluate(nominal(s, 0.0)).Fx0);
        peak_y = std::min(peak_y, tyre.evaluate(nominal(0.0, s)).Fy0);   // Kya < 0: the peak is negative
    }
    const Real Dx = p.PDX1 * 4000.0, SVx = 4000.0 * p.PVX1;
    const Real Dy = p.PDY1 * 4000.0, SVy = 4000.0 * p.PVY1;
    INFO("Fx0 peak " << peak_x << " N against " << Dx + SVx << "; Fy0 peak " << peak_y << " against " << -Dy + SVy);
    CHECK_THAT(peak_x, WithinRel(Dx + SVx, 1e-5));
    CHECK(peak_x <= Dx + SVx);
    CHECK_THAT(peak_y, WithinRel(-Dy + SVy, 1e-5));
    CHECK(peak_y >= -Dy + SVy);
}

TEST_CASE("Magic Formula: combined slip leaves each force alone where the other slip is zero", "[tyre][mf]")
{
    // (4.E51): Gxa is the cosine form at alpha_S = alpha + SHxa over the
    // same at SHxa: at alpha = 0 the two are one number, so Fx = Fx0
    // exactly. (4.E59) likewise makes Gyk = 1 at kappa = 0, and (4.E66)
    // SVyk = DVyk sin(rVy5 atan(0)) = 0: Fy = Fy0 exactly. Both tyres, both
    // versions' formulas.
    for (const char* file : {"pacejka_a3_205_60R15.tir", "mf52_test_tyre.tir"}) {
        INFO(file);
        const MagicFormulaTyre tyre(io::read_tir(data(file)).parameters);
        for (const Real s : {-0.3, -0.05, 0.02, 0.1, 0.5}) {
            const MagicFormulaOutput x = tyre.evaluate(nominal(s, 0.0));
            CHECK(x.Fx == x.Fx0);
            const MagicFormulaOutput y = tyre.evaluate(nominal(0.0, s));
            CHECK(y.Fy == y.Fy0);
        }
    }
}

TEST_CASE("Magic Formula: rolling resistance and relaxation lengths at the book's nominal state", "[tyre][mf]")
{
    // The paper's (50) at Fz = Fzo, p = pio, no inclination and no Fx: My =
    // -Ro Fzo (qsy1 + qsy3 |Vx / Vo| + qsy4 (Vx / Vo)^4), against the
    // rolling (about +y, forwards), and at Vx = Vo = 16.7 m/s -4000 x 0.3135
    // x (0.00702 + 0.001515 + 8.514e-5) = -10.81 N m. The
    // relaxation lengths of MF 6.1 are the slip stiffnesses over the
    // carcass's stiffnesses (cx, cy of [structural] at dfz = dpi = 0):
    // sigma_kappa = 86748 / 358066 = 0.2423 m, sigma_alpha = |Kya| / 102673.
    // To rounding.
    const MagicFormulaTyre tyre = appendix3();
    const MagicFormulaParameters& p = tyre.parameters();
    // Fx = 0 where the curve crosses zero; at kappa = -SHx it is SVx, 0.08 N,
    // whose qsy2 term is 0 here anyway (qsy2 = 0).
    const MagicFormulaOutput o = tyre.evaluate(nominal(-p.PHX1, 0.0));
    CHECK_THAT(o.My, WithinRel(-4000.0 * 0.3135 * (0.00702 + 0.001515 + 8.514e-5), 1e-13));
    CHECK_THAT(o.sigma_kappa, WithinRel(4000.0 * p.PKX1 / p.LONGITUDINAL_STIFFNESS, 1e-13));
    CHECK_THAT(o.sigma_alpha, WithinRel(std::abs(o.Kya) / p.LATERAL_STIFFNESS, 1e-13));
}

TEST_CASE("Magic Formula: the library's Appendix 3 tyre is the file's", "[tyre][mf][tir]")
{
    // pacejka_appendix3_tyre(), the vehicles' default (task 7.4), against the
    // file written from the same table: every parameter equal, bit for bit
    // (both are the table's decimals read as doubles).
    const MagicFormulaParameters a = pacejka_appendix3_tyre();
    const MagicFormulaParameters b = io::read_tir(data("pacejka_a3_205_60R15.tir")).parameters;
    CHECK(a.version == b.version);
    CHECK(a.FITTYP == b.FITTYP);
    CHECK(a.side == b.side);
#define MBD_SAME(section, name, value) CHECK(a.name == b.name);
    MBD_MAGIC_FORMULA_PARAMETERS(MBD_SAME)
#undef MBD_SAME
}

TEST_CASE("Magic Formula: the reader takes TYRESIDE", "[tyre][mf][tir]")
{
    const std::string base = "[MODEL]\nFITTYP = 61\n[DIMENSION]\nUNLOADED_RADIUS = 0.3\n[VERTICAL]\nFNOMIN = 4000\n";
    CHECK(io::parse_tir(base).parameters.side == TyreSide::Left);
    CHECK(io::parse_tir("[MODEL]\nTYRESIDE = 'RIGHT'\n" + base.substr(8)).parameters.side == TyreSide::Right);
    CHECK(io::parse_tir("[MODEL]\nTYRESIDE = 'Left'\n" + base.substr(8)).parameters.side == TyreSide::Left);
    CHECK_THROWS_WITH(io::parse_tir("[MODEL]\nTYRESIDE = 'MIDDLE'\n" + base.substr(8)),
                      ContainsSubstring("MBD-I024"));
}

TEST_CASE("Magic Formula: a road's friction is LMUX and LMUY scaled", "[tyre][mf]")
{
    // MagicFormulaInput::friction multiplies LMUX and LMUY, and nothing else:
    // at a friction of 0.6 the outputs are those of the set with both scaling
    // factors 0.6 times theirs, to the last bit at a combined slip.
    MagicFormulaParameters scaled = io::read_tir(data("pacejka_a3_205_60R15.tir")).parameters;
    const MagicFormulaTyre tyre(scaled);
    scaled.LMUX *= 0.6;
    scaled.LMUY *= 0.6;
    const MagicFormulaTyre slippery(scaled);
    MagicFormulaInput in = nominal(-0.07, 0.04);
    in.gamma = 0.02;
    const MagicFormulaOutput reference = slippery.evaluate(in);
    in.friction = 0.6;
    const MagicFormulaOutput o = tyre.evaluate(in);
    CHECK(o.Fx == reference.Fx);
    CHECK(o.Fy == reference.Fy);
    CHECK(o.Mz == reference.Mz);
    CHECK(o.Mx == reference.Mx);
}

TEST_CASE("Magic Formula: reduced to the simple formula, the stiffnesses and friction are kept", "[tyre][mf]")
{
    // PacejkaTireParams::FromMagicFormula, for the bicycle model of a vehicle
    // (task 7.4). At the nominal load the simple tyre's slip stiffnesses are
    // the Magic Formula's (|Kya| positive), to rounding; its friction is
    // pDx1 + pDx2 dfz and pDy1 + pDy2 dfz, as the Magic Formula's upright,
    // at every load (both linear in dfz); its stiffness's slope in dfz is the
    // derivative of the Magic Formula's at dfz = 0: central differences of
    // 1e-4 in dfz, truncation (1e-4)^2 times the third derivative (order
    // 1e5 N here) and rounding 1e-16 x 1e5 / 1e-4: allowed 1e-6 relative.
    const MagicFormulaParameters p = io::read_tir(data("pacejka_a3_205_60R15.tir")).parameters;
    const MagicFormulaTyre mf(p);
    const PacejkaTire simple(PacejkaTireParams::FromMagicFormula(p));
    const MagicFormulaOutput c = mf.characteristics(4000.0, 0.0);
    CHECK_THAT(simple.cornering_stiffness(4000.0), WithinRel(-c.Kya, 1e-14));
    CHECK_THAT(simple.longitudinal_stiffness(4000.0), WithinRel(c.Kxk, 1e-14));
    for (const Real Fz : {2500.0, 4000.0, 6500.0}) {
        const MagicFormulaOutput at = mf.characteristics(Fz, 0.0);
        CHECK_THAT(simple.peak_mu_lateral(Fz), WithinRel(at.muy, 1e-14));
        CHECK_THAT(simple.peak_mu_longitudinal(Fz), WithinRel(at.mux, 1e-14));
    }
    const Real h = 1e-4;
    const Real dKya = (mf.characteristics(4000.0 * (1 + h), 0.0).Kya - mf.characteristics(4000.0 * (1 - h), 0.0).Kya)
                    / (2 * h);
    const Real dKxk = (mf.characteristics(4000.0 * (1 + h), 0.0).Kxk - mf.characteristics(4000.0 * (1 - h), 0.0).Kxk)
                    / (2 * h);
    const PacejkaTireParams sp = PacejkaTireParams::FromMagicFormula(p);
    CHECK_THAT(sp.lateral.K_Fz, WithinRel(-dKya, 1e-6));
    CHECK_THAT(sp.longitudinal.K_Fz, WithinRel(dKxk, 1e-6));
}
