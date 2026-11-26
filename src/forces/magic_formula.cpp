#include "mbd/forces/magic_formula.hpp"

#include <algorithm>
#include <cmath>

// Equation numbers: (4.En) are Pacejka's, 3rd ed. (2012), section 4.3.2; (n)
// alone are those of the appendix of Besselink, Schmeitz and Pacejka (2010),
// TNO's MF-Tyre 6.1. Where both are given the two agree; where only the
// paper's is, the paper is followed (magic_formula.hpp, docs/tyres.md).

namespace mbd {

namespace {

/// The small quantity epsilon that keeps the denominators of (4.E16),
/// (4.E26), (4.E27) and (4.E39) from zero, given its neighbour's sign.
constexpr Real kEpsilon = 1e-6;

Real sgn(Real x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

/// Plus epsilon with x's sign (positive at zero).
Real guard(Real x) { return x + (x < 0.0 ? -kEpsilon : kEpsilon); }

/// The Magic Formula's sine form, D sin(C atan(B x - E (B x - atan(B x)))).
Real sine_form(Real B, Real C, Real D, Real E, Real x)
{
    const Real Bx = B * x;
    return D * std::sin(C * std::atan(Bx - E * (Bx - std::atan(Bx))));
}

/// Its cosine form, cos(C atan(B x - E (B x - atan(B x)))).
Real cosine_form(Real B, Real C, Real E, Real x)
{
    const Real Bx = B * x;
    return std::cos(C * std::atan(Bx - E * (Bx - std::atan(Bx))));
}

/// The pure-slip side force and the quantities the aligning torque needs,
/// at one inclination.
struct Lateral {
    Real Fy0, Kya, SHy, SVy, By, Cy, muy;
};

} // namespace

std::string to_string(MagicFormulaVersion version)
{
    return version == MagicFormulaVersion::MF52 ? "MF 5.2" : "MF 6.1";
}

MagicFormulaTyre::MagicFormulaTyre(const MagicFormulaParameters& parameters)
    : p_(parameters)
{
    auto positive = [](Real v, const char* name) {
        MBD_THROW_IF(!(v > 0.0), std::string("MBD-F053: Magic Formula tyre: ") + name + " must be > 0; it is "
                                     + std::to_string(v));
    };
    positive(p_.FNOMIN, "FNOMIN");
    positive(p_.UNLOADED_RADIUS, "UNLOADED_RADIUS");
    positive(p_.LONGVL, "LONGVL");
    positive(p_.LFZO, "LFZO");
    if (p_.version == MagicFormulaVersion::MF61) positive(p_.NOMPRES, "NOMPRES");
}

MagicFormulaOutput MagicFormulaTyre::evaluate(const MagicFormulaInput& in) const
{
    const MagicFormulaParameters& p = p_;
    const bool v61 = p.version == MagicFormulaVersion::MF61;
    MagicFormulaOutput out;
    const Real Fz = in.Fz;
    out.Fz = std::max(Fz, Real(0.0));
    if (!(Fz > 0.0)) return out;   // off the road

    // --- Common quantities ---
    const Real Fz0 = p.FNOMIN;
    const Real Fz0p = p.LFZO * Fz0;                                           // (4.E1)
    const Real dfz = (Fz - Fz0p) / Fz0p;                                      // (4.E2a), (31)
    const Real pressure = in.pressure > 0.0 ? in.pressure : p.INFLPRES;
    const Real dpi = v61 ? (pressure - p.NOMPRES) / p.NOMPRES : 0.0;         // (4.E2b), (32); MF 5.2 has none
    const Real R0 = p.UNLOADED_RADIUS;
    const Real Vcx = in.Vcx;
    const Real sVcx = Vcx < 0.0 ? -1.0 : 1.0;
    const Real kappa = in.kappa;
    const Real alpha = in.alpha;
    const Real cos_alpha = std::cos(alpha);                                   // cos(alpha_M) of (83), (88)
    // The inclination as the forces and moments take it: MF 5.2 scales it
    // for each (lambda_gamma x, y, z); MF 6.1 scales the camber stiffnesses
    // instead (LKYC, LKZC).
    const Real gamma = in.gamma;
    const Real gx = v61 ? gamma : gamma * p.LGAX;
    const Real gy = v61 ? gamma : gamma * p.LGAY;
    const Real gz = v61 ? gamma : gamma * p.LGAZ;
    // Friction falling with slip speed (4.E7), and the road's friction on
    // both. The degressive form (4.E8) is taken with A_mu = 1, lambda' =
    // lambda*, as TNO does (D63).
    const Real Vs = std::hypot(kappa * Vcx, std::tan(alpha) * Vcx);
    const Real LMUX_s = in.friction * p.LMUX / (1.0 + p.LMUV * Vs / p.LONGVL);
    const Real LMUY_s = in.friction * p.LMUY / (1.0 + p.LMUV * Vs / p.LONGVL);
    // TNO's low-speed model, as MFeval has it: below VXLOW the shifts (SHx,
    // SVx, SHy, SVy, SVyk), the trail and the residual torque fade out, to
    // none at rest, by (1 - cos(pi |Vcx| / VXLOW)) / 2 (D64).
    const Real speed = std::abs(Vcx);
    const Real fade = p.VXLOW > 0.0 && speed < p.VXLOW ? 0.5 * (1.0 - std::cos(pi * speed / p.VXLOW)) : 1.0;

    // --- Longitudinal force, pure longitudinal slip ---
    const Real SHx = (p.PHX1 + p.PHX2 * dfz) * p.LHX * fade;                  // (4.E17), (41)
    const Real kx = kappa + SHx;                                              // (4.E10), (34)
    const Real Cx = p.PCX1 * p.LCX;                                           // (4.E11), (35)
    const Real mux = (p.PDX1 + p.PDX2 * dfz) * (1.0 + p.PPX3 * dpi + p.PPX4 * dpi * dpi)
                   * (1.0 - p.PDX3 * gx * gx) * LMUX_s;                       // (4.E13), (37)
    const Real Dx = mux * Fz;                                                 // (4.E12), (36)
    const Real Ex = std::min(Real(1.0), (p.PEX1 + p.PEX2 * dfz + p.PEX3 * dfz * dfz)
                                            * (1.0 - p.PEX4 * sgn(kx)) * p.LEX);   // (4.E14), (38)
    const Real Kxk = Fz * (p.PKX1 + p.PKX2 * dfz) * std::exp(p.PKX3 * dfz)
                   * (1.0 + p.PPX1 * dpi + p.PPX2 * dpi * dpi) * p.LKX;       // (4.E15), (39)
    const Real Bx = Kxk / guard(Cx * Dx);                                     // (4.E16), (40)
    const Real SVx = Fz * (p.PVX1 + p.PVX2 * dfz) * p.LVX * LMUX_s * fade;    // (4.E18), (42)
    const Real Fx0 = sine_form(Bx, Cx, Dx, Ex, kx) + SVx;                     // (4.E9), (33)

    // --- Side force, pure side slip, at an inclination g ---
    auto lateral = [&](Real g) {
        Lateral l{};
        l.Kya = p.PKY1 * Fz0p * (1.0 + p.PPY1 * dpi) * (1.0 - p.PKY3 * std::abs(g))
              * std::sin(p.PKY4 * std::atan((Fz / Fz0p) / ((p.PKY2 + p.PKY5 * g * g) * (1.0 + p.PPY2 * dpi))))
              * p.LKY;                                                        // (4.E25), (58)
        const Real SVyg = Fz * (p.PVY3 + p.PVY4 * dfz) * g * p.LKYC * LMUY_s; // (4.E28), (66)
        const Real SHy0 = (p.PHY1 + p.PHY2 * dfz) * p.LHY;                    // (62)
        if (v61) {
            const Real Kyg0 = Fz * (p.PKY6 + p.PKY7 * dfz) * (1.0 + p.PPY5 * dpi) * p.LKYC;   // (4.E30), (59)
            l.SHy = SHy0 + (Kyg0 * g - SVyg) / guard(l.Kya);                  // (4.E27), (61), (63)
        } else {
            l.SHy = SHy0 + p.PHY3 * g;                                        // MF 5.2: a shift in proportion
        }
        l.SVy = Fz * (p.PVY1 + p.PVY2 * dfz) * p.LVY * LMUY_s + SVyg;         // (4.E29), (64), (65)
        l.SHy *= fade;
        l.SVy *= fade;
        l.muy = (p.PDY1 + p.PDY2 * dfz) * (1.0 + p.PPY3 * dpi + p.PPY4 * dpi * dpi)
              * (1.0 - p.PDY3 * g * g) * LMUY_s;                              // (4.E23), (56)
        const Real ay = alpha + l.SHy;                                        // (4.E20), (53)
        l.Cy = p.PCY1 * p.LCY;                                                // (4.E21), (54)
        const Real Dy = l.muy * Fz;                                           // (4.E22), (55)
        const Real Ey = std::min(Real(1.0), (p.PEY1 + p.PEY2 * dfz)
                                                * (1.0 + p.PEY5 * g * g - (p.PEY3 + p.PEY4 * g) * sgn(ay))
                                                * p.LEY);                     // (4.E24), (57)
        l.By = l.Kya / guard(l.Cy * Dy);                                      // (4.E26), (60)
        l.Fy0 = sine_form(l.By, l.Cy, Dy, Ey, ay) + l.SVy;                    // (4.E19), (52)
        return l;
    };
    const Lateral lat = lateral(gy);
    const Real Fy0 = lat.Fy0;
    // MF 6.1's aligning torque takes the side force at zero inclination
    // ((4.E32), (75)); MF 5.2's the side force itself.
    const Real Fy0_upright = v61 ? lateral(0.0).Fy0 : Fy0;

    // --- Aligning torque: trail and residual torque ---
    const Real SHt = p.QHZ1 + p.QHZ2 * dfz + (p.QHZ3 + p.QHZ4 * dfz) * gz;   // (4.E35), (77)
    const Real at = alpha + SHt;                                              // (4.E34), (76)
    const Real Kya_p = guard(lat.Kya);                                        // (4.E39)
    const Real ar = alpha + lat.SHy + lat.SVy / Kya_p;                        // (4.E37), (4.E38), (78)
    const Real Bt = (p.QBZ1 + p.QBZ2 * dfz + p.QBZ3 * dfz * dfz) * (1.0 + p.QBZ4 * gz + p.QBZ5 * std::abs(gz))
                  * p.LKY / LMUY_s;                                           // (84)
    const Real Ct = p.QCZ1;                                                   // (4.E41), (85)
    // The paper's Dt (86) with the book's sign of the forward speed (4.E42).
    const Real Dt = Fz * (R0 / Fz0) * (p.QDZ1 + p.QDZ2 * dfz) * (1.0 - p.PPZ1 * dpi)
                  * (1.0 + p.QDZ3 * gz + p.QDZ4 * gz * gz) * p.LTR * sVcx;   // (86)
    const Real Et = std::min(Real(1.0), (p.QEZ1 + p.QEZ2 * dfz + p.QEZ3 * dfz * dfz)
                                            * (1.0 + (p.QEZ4 + p.QEZ5 * gz) * (2.0 / pi) * std::atan(Bt * Ct * at)));   // (4.E44), (87)
    const Real Br = p.QBZ9 * p.LKY / LMUY_s + p.QBZ10 * lat.By * lat.Cy;      // (4.E45), (89)
    // The paper's Dr (90), without cos(alpha) (it is in Mzr), with the book's
    // sign of the forward speed (4.E47).
    const Real Dr = Fz * R0
                  * ((p.QDZ6 + p.QDZ7 * dfz) * p.LRES
                     + ((p.QDZ8 + p.QDZ9 * dfz) * (1.0 + p.PPZ2 * dpi) + (p.QDZ10 + p.QDZ11 * dfz) * std::abs(gz)) * gz
                           * p.LKZC)
                  * LMUY_s * sVcx;                                            // (90)
    // Pure side slip ((79): the equivalent slips are the slips).
    const Real t0 = Dt * cosine_form(Bt, Ct, Et, at) * cos_alpha * fade;      // (83)
    const Real Mzr0 = Dr * std::cos(std::atan(Br * ar)) * cos_alpha * fade;   // (88)
    const Real Mz0 = -t0 * Fy0_upright + Mzr0;                                // (4.E31), (75)

    // --- Longitudinal force, combined slip ---
    const Real Bxa = (p.RBX1 + p.RBX3 * gx * gx) * std::cos(std::atan(p.RBX2 * kappa)) * p.LXAL;   // (4.E54), (45)
    const Real Cxa = p.RCX1;                                                  // (4.E55), (46)
    const Real Exa = std::min(Real(1.0), p.REX1 + p.REX2 * dfz);              // (4.E56), (47)
    const Real SHxa = p.RHX1;                                                 // (4.E57), (48)
    const Real as = alpha + SHxa;                                             // (4.E53), (44)
    const Real Gxa = cosine_form(Bxa, Cxa, Exa, as) / cosine_form(Bxa, Cxa, Exa, SHxa);   // (4.E51), (4.E52), (43)
    const Real Fx = Gxa * Fx0;                                                // (4.E50), (33)

    // --- Side force, combined slip ---
    const Real Byk = (p.RBY1 + p.RBY4 * gy * gy) * std::cos(std::atan(p.RBY2 * (alpha - p.RBY3))) * p.LYKA;   // (4.E62), (71)
    const Real Cyk = p.RCY1;                                                  // (4.E63), (72)
    const Real Eyk = std::min(Real(1.0), p.REY1 + p.REY2 * dfz);              // (4.E64), (73)
    const Real SHyk = p.RHY1 + p.RHY2 * dfz;                                  // (4.E65), (74)
    const Real ks = kappa + SHyk;                                             // (4.E61), (70)
    const Real Gyk = cosine_form(Byk, Cyk, Eyk, ks) / cosine_form(Byk, Cyk, Eyk, SHyk);   // (4.E59), (4.E60), (69)
    const Real DVyk = lat.muy * Fz * (p.RVY1 + p.RVY2 * dfz + p.RVY3 * gy) * std::cos(std::atan(p.RVY4 * alpha));   // (4.E67), (68)
    const Real SVyk = DVyk * std::sin(p.RVY5 * std::atan(p.RVY6 * kappa)) * p.LVYKA * fade;   // (4.E66), (67)
    const Real Fy = Gyk * Fy0 + SVyk;                                         // (4.E58), (51)

    // --- Aligning torque, combined slip ---
    const Real k_eq = Kxk / Kya_p * kappa;
    const Real at_eq = std::atan(std::sqrt(std::tan(at) * std::tan(at) + k_eq * k_eq)) * sgn(at);   // (80)
    const Real ar_eq = std::atan(std::sqrt(std::tan(ar) * std::tan(ar) + k_eq * k_eq)) * sgn(ar);   // (81)
    const Real s = R0 * (p.SSZ1 + p.SSZ2 * (Fy / Fz0) + (p.SSZ3 + p.SSZ4 * dfz) * gz) * p.LS;   // (82)
    const Real t = Dt * cosine_form(Bt, Ct, Et, at_eq) * cos_alpha * fade;   // (83)
    const Real Mzr = Dr * std::cos(std::atan(Br * ar_eq)) * cos_alpha * fade; // (88)
    // MF 6.1: the combined side force at zero inclination, Gyk (at the
    // inclination, as TNO's solver has it, D63) times Fy0 at zero inclination
    // (75); MF 5.2: the side force less its slip-induced part.
    const Real Fy_arm = v61 ? Gyk * Fy0_upright : Fy - SVyk;
    const Real Mz = -t * Fy_arm + Mzr + s * Fx;                               // (4.E71), (75)

    // --- Overturning couple (49); MF 5.2's is its first three terms ---
    const Real Mx = R0 * Fz * p.LMX
                      * (p.QSX1 * p.LVMX - p.QSX2 * gamma * (1.0 + p.PPMX1 * dpi) - p.QSX12 * gamma * std::abs(gamma)
                         + p.QSX3 * Fy / Fz0
                         + p.QSX4 * std::cos(p.QSX5 * std::atan(std::pow(p.QSX6 * Fz / Fz0, 2.0)))
                               * std::sin(p.QSX7 * gamma + p.QSX8 * std::atan(p.QSX9 * Fy / Fz0))
                         + p.QSX10 * std::atan(p.QSX11 * Fz / Fz0) * gamma)
                  + R0 * Fy * p.LMX * (p.QSX13 + p.QSX14 * std::abs(gamma));

    // --- Rolling resistance moment: (50) for MF 6.1, R0 Fz (...) for MF 5.2;
    // against the rolling ---
    const Real V0 = p.LONGVL;
    const Real speed_terms = p.QSY1 + p.QSY2 * Fx / Fz0 + p.QSY3 * std::abs(Vcx / V0) + p.QSY4 * std::pow(Vcx / V0, 4.0);
    const Real My = -sVcx
                  * (v61 ? R0 * Fz0 * p.LMY * (speed_terms + (p.QSY5 + p.QSY6 * Fz / Fz0) * gamma * gamma)
                               * std::pow(Fz / Fz0, p.QSY7) * std::pow(pressure / p.NOMPRES, p.QSY8)
                         : R0 * Fz * p.LMY * speed_terms);

    // --- Relaxation lengths ---
    if (v61) {
        // The slip stiffnesses over the carcass's stiffnesses (paper, (17) to
        // (20) of its text).
        const Real cx = p.LONGITUDINAL_STIFFNESS * (1.0 + p.PCFX1 * dfz + p.PCFX2 * dfz * dfz) * (1.0 + p.PCFX3 * dpi);
        const Real cy = p.LATERAL_STIFFNESS * (1.0 + p.PCFY1 * dfz + p.PCFY2 * dfz * dfz) * (1.0 + p.PCFY3 * dpi);
        out.sigma_kappa = cx > 0.0 ? std::abs(Kxk / cx) : 0.0;
        out.sigma_alpha = cy > 0.0 ? std::abs(lat.Kya / cy) : 0.0;
    } else {
        out.sigma_kappa = Fz * (p.PTX1 + p.PTX2 * dfz) * std::exp(-p.PTX3 * dfz) * (R0 / Fz0) * p.LSGKP;
        out.sigma_alpha = p.PTY1 * std::sin(2.0 * std::atan(Fz / (p.PTY2 * Fz0p))) * (1.0 - p.PKY3 * std::abs(gy)) * R0
                        * p.LFZO * p.LSGAL;
    }

    out.Fx = Fx;
    out.Fy = Fy;
    out.Mx = Mx;
    out.My = My;
    out.Mz = Mz;
    out.Fx0 = Fx0;
    out.Fy0 = Fy0;
    out.Mz0 = Mz0;
    out.Kxk = Kxk;
    out.Kya = lat.Kya;
    out.mux = mux;
    out.muy = lat.muy;
    out.trail = t;
    out.Mzr = Mzr;
    out.s = s;
    return out;
}

MagicFormulaOutput MagicFormulaTyre::characteristics(Real Fz, Real gamma, Real pressure, Real friction) const
{
    MagicFormulaInput in;
    in.Fz = Fz;
    in.gamma = gamma;
    in.pressure = pressure;
    in.friction = friction;
    in.kappa = 0.0;
    in.alpha = 0.0;
    in.Vcx = 0.0;
    return evaluate(in);
}

MagicFormulaParameters pacejka_appendix3_tyre()
{
    // Every value of Table A3.1 that is not 0, as the file has them; the
    // others (Q_FCX, Q_FCY, PCFX2, PCFX3, PCFY2, PDX3, PEX3, RBX3, QSX12 to
    // QSX14, PPMX1, PDY3, PEY5, PKY5, PPY5, RBY4, RVY3, QSY2, QSY5, QSY6,
    // QBZ3, QBZ10, QDZ4, QDZ10, QDZ11, QEZ3, PPZ2, SSZ3, SSZ4) are 0, as their
    // defaults. Every scaling factor is 1.
    MagicFormulaParameters p;
    p.version = MagicFormulaVersion::MF61;
    p.FITTYP = 61;
    p.LONGVL = 16.7; p.VXLOW = 1; p.UNLOADED_RADIUS = 0.3135; p.WIDTH = 0.205; p.RIM_RADIUS = 0.1905;
    p.INFLPRES = 220000; p.NOMPRES = 220000; p.FNOMIN = 4000; p.VERTICAL_STIFFNESS = 209651;
    p.VERTICAL_DAMPING = 50; p.Q_V2 = 0.04667; p.Q_FZ2 = 15.4; p.PFZ1 = 0.7098; p.BREFF = 8.386;
    p.DREFF = 0.25826; p.FREFF = 0.07394; p.Q_RE0 = 0.9974; p.Q_V1 = 7.742e-4;
    p.LONGITUDINAL_STIFFNESS = 358066; p.LATERAL_STIFFNESS = 102673; p.PCFX1 = 0.17504; p.PCFY1 = 0.16365;
    p.PCFY3 = 0.24993; p.PCX1 = 1.579; p.PDX1 = 1.0422; p.PDX2 = -0.08285; p.PEX1 = 0.11113; p.PEX2 = 0.3143;
    p.PEX4 = 0.001719; p.PKX1 = 21.687; p.PKX2 = 13.728; p.PKX3 = -0.4098; p.PHX1 = 2.1615e-4;
    p.PHX2 = 0.0011598; p.PVX1 = 2.0283e-5; p.PVX2 = 1.0568e-4; p.PPX1 = -0.3485; p.PPX2 = 0.37824;
    p.PPX3 = -0.09603; p.PPX4 = 0.06518; p.RBX1 = 13.046; p.RBX2 = 9.718; p.RCX1 = 0.9995; p.REX1 = -0.4403;
    p.REX2 = -0.4663; p.RHX1 = -9.968e-5; p.QSX1 = -0.007764; p.QSX2 = 1.1915; p.QSX3 = 0.013948;
    p.QSX4 = 4.912; p.QSX5 = 1.02; p.QSX6 = 22.83; p.QSX7 = 0.7104; p.QSX8 = -0.023393; p.QSX9 = 0.6581;
    p.QSX10 = 0.2824; p.QSX11 = 5.349; p.PCY1 = 1.338; p.PDY1 = 0.8785; p.PDY2 = -0.06452; p.PEY1 = -0.8057;
    p.PEY2 = -0.6046; p.PEY3 = 0.09854; p.PEY4 = -6.697; p.PKY1 = -15.324; p.PKY2 = 1.715; p.PKY3 = 0.3695;
    p.PKY4 = 2.0005; p.PKY6 = -0.8987; p.PKY7 = -0.23303; p.PHY1 = -0.001806; p.PHY2 = 0.00352;
    p.PVY1 = -0.00661; p.PVY2 = 0.03592; p.PVY3 = -0.162; p.PVY4 = -0.4864; p.PPY1 = -0.6255;
    p.PPY2 = -0.06523; p.PPY3 = -0.16666; p.PPY4 = 0.2811; p.RBY1 = 10.622; p.RBY2 = 7.82; p.RBY3 = 0.002037;
    p.RCY1 = 1.0587; p.REY1 = 0.3148; p.REY2 = 0.004867; p.RHY1 = 0.009472; p.RHY2 = 0.009754;
    p.RVY1 = 0.05187; p.RVY2 = 4.853e-4; p.RVY4 = 94.63; p.RVY5 = 1.8914; p.RVY6 = 23.8; p.QSY1 = 0.00702;
    p.QSY3 = 0.001515; p.QSY4 = 8.514e-5; p.QSY7 = 0.9008; p.QSY8 = -0.4089; p.QBZ1 = 12.035; p.QBZ2 = -1.33;
    p.QBZ4 = 0.176; p.QBZ5 = -0.14853; p.QBZ9 = 34.5; p.QCZ1 = 1.2923; p.QDZ1 = 0.09068; p.QDZ2 = -0.00565;
    p.QDZ3 = 0.3778; p.QDZ6 = 0.0017015; p.QDZ7 = -0.002091; p.QDZ8 = -0.1428; p.QDZ9 = 0.00915;
    p.QEZ1 = -1.7924; p.QEZ2 = 0.8975; p.QEZ4 = 0.2895; p.QEZ5 = -0.6786; p.QHZ1 = 0.0014333;
    p.QHZ2 = 0.0024087; p.QHZ3 = 0.24973; p.QHZ4 = -0.21205; p.PPZ1 = -0.4408; p.SSZ1 = 0.00918;
    p.SSZ2 = 0.03869;
    return p;
}

} // namespace mbd
