#include "mbd/forces/pacejka.hpp"

namespace mbd {

PacejkaTireParams PacejkaTireParams::FromMagicFormula(const MagicFormulaParameters& m)
{
    PacejkaTireParams p = DefaultPassengerCar();
    const Real Fz0 = m.LFZO * m.FNOMIN;

    // Longitudinal: Kxk = Fz (pKx1 + pKx2 dfz) exp(pKx3 dfz) lKx (4.E15),
    // with Fz = Fz0 (1 + dfz): Fz0 pKx1 lKx at dfz = 0, its slope there Fz0
    // (pKx1 + pKx2 + pKx1 pKx3) lKx.
    p.longitudinal.Fz0 = Fz0;
    p.longitudinal.mu0 = m.PDX1 * m.LMUX;
    p.longitudinal.mu_Fz = m.PDX2 * m.LMUX;
    p.longitudinal.C = m.PCX1 * m.LCX;
    p.longitudinal.K0 = Fz0 * m.PKX1 * m.LKX;
    p.longitudinal.K_Fz = Fz0 * (m.PKX1 + m.PKX2 + m.PKX1 * m.PKX3) * m.LKX;
    p.longitudinal.E0 = m.PEX1 * m.LEX;
    p.longitudinal.E_Fz = m.PEX2 * m.LEX;

    // Lateral, upright and at the nominal pressure: Kya = pKy1 Fz0 sin(pKy4
    // atan((1 + dfz) / pKy2)) lKy (4.E25); d atan((1 + dfz) / pKy2) / d dfz
    // at 0 is x / (1 + x^2), x = 1 / pKy2.
    const Real x = 1.0 / m.PKY2;
    const Real Kya = m.PKY1 * Fz0 * std::sin(m.PKY4 * std::atan(x)) * m.LKY;
    const Real slope = m.PKY1 * Fz0 * std::cos(m.PKY4 * std::atan(x)) * m.PKY4 * x / (1.0 + x * x) * m.LKY;
    const Real sign = Kya < 0.0 ? -1.0 : 1.0;   // ISO's stiffness is negative
    p.lateral.Fz0 = Fz0;
    p.lateral.mu0 = m.PDY1 * m.LMUY;
    p.lateral.mu_Fz = m.PDY2 * m.LMUY;
    p.lateral.C = m.PCY1 * m.LCY;
    p.lateral.K0 = sign * Kya;
    p.lateral.K_Fz = sign * slope;
    p.lateral.E0 = m.PEY1 * m.LEY;
    p.lateral.E_Fz = m.PEY2 * m.LEY;

    p.Gx_alpha.B_comb = m.RBX1 * m.LXAL;
    p.Gx_alpha.C_comb = m.RCX1;
    p.Gx_alpha.D_comb = 1.0;
    p.Gy_kappa.B_comb = m.RBY1 * m.LYKA;
    p.Gy_kappa.C_comb = m.RCY1;
    p.Gy_kappa.D_comb = 1.0;
    return p;
}

PacejkaTireParams PacejkaTireParams::DefaultPassengerCar()
{
    PacejkaTireParams p;

    // Lateral (Fy vs alpha)
    p.lateral.mu0    = 1.0;
    p.lateral.mu_Fz  = -0.001;
    p.lateral.C      = 1.3;
    p.lateral.K0     = 45000.0;
    p.lateral.K_Fz   = 0.0;
    p.lateral.E0     = -1.5;
    p.lateral.E_Fz   = 0.0;
    p.lateral.Fz0    = 4000.0;

    // Longitudinal (Fx vs kappa)
    p.longitudinal.mu0    = 1.1;
    p.longitudinal.mu_Fz  = -0.001;
    p.longitudinal.C      = 1.65;
    p.longitudinal.K0     = 60000.0;
    p.longitudinal.K_Fz   = 0.0;
    p.longitudinal.E0     = -0.5;
    p.longitudinal.E_Fz   = 0.0;
    p.longitudinal.Fz0    = 4000.0;

    // Combined slip weighting
    p.Gx_alpha.B_comb = 8.0;
    p.Gx_alpha.C_comb = 1.1;
    p.Gx_alpha.D_comb = 1.0;

    p.Gy_kappa.B_comb = 6.0;
    p.Gy_kappa.C_comb = 1.2;
    p.Gy_kappa.D_comb = 1.0;

    return p;
}

Real PacejkaTire::evaluate_magic_formula(Real x,
                                         Real Fz,
                                         const MagicFormulaCoeffs& c,
                                         Real friction)
{
    const Real dFz = (Fz - c.Fz0) / c.Fz0;

    // Peak factor, scaled by the road's friction
    const Real mu = c.mu0 + c.mu_Fz * dFz;
    const Real D  = mu * Fz * friction;

    // Stiffness
    const Real K = c.K0 + c.K_Fz * dFz;

    // Shape factor
    const Real C = c.C;

    // Stiffness factor: B = K / (C * D)
    const Real CD = C * D;
    const Real B = (std::abs(CD) > Real(1e-6)) ? K / CD : Real(0.0);

    // Curvature factor
    const Real E = c.E0 + c.E_Fz * dFz;

    // Magic Formula
    const Real Bx = B * x;
    const Real inner = Bx - E * (Bx - std::atan(Bx));
    return D * std::sin(C * std::atan(inner));
}

Real PacejkaTire::evaluate_combined_weight(Real x_other,
                                           const CombinedSlipCoeffs& c)
{
    const Real Bx = c.B_comb * x_other;
    return c.D_comb * std::cos(c.C_comb * std::atan(Bx));
}

TireForceResult PacejkaTire::compute(Real kappa, Real alpha, Real Fz, Real friction) const
{
    TireForceResult r;
    r.kappa = kappa;
    r.alpha = alpha;
    r.Fz    = Fz;

    if (Fz <= Real(0.0)) {
        return r; // No contact
    }

    // Pure slip forces
    r.Fx_pure = evaluate_magic_formula(kappa, Fz, params.longitudinal, friction);
    r.Fy_pure = evaluate_magic_formula(alpha, Fz, params.lateral, friction);

    // Combined slip weighting
    r.Gx = evaluate_combined_weight(alpha, params.Gx_alpha);
    r.Gy = evaluate_combined_weight(kappa, params.Gy_kappa);

    // Clamp weighting to [0, 1] for physical plausibility
    r.Gx = std::clamp(r.Gx, Real(0.0), Real(1.0));
    r.Gy = std::clamp(r.Gy, Real(0.0), Real(1.0));

    r.Fx = r.Fx_pure * r.Gx;
    r.Fy = r.Fy_pure * r.Gy;

    return r;
}

Real PacejkaTire::cornering_stiffness(Real Fz) const
{
    const auto& c = params.lateral;
    const Real dFz = (Fz - c.Fz0) / c.Fz0;
    return c.K0 + c.K_Fz * dFz;
}

Real PacejkaTire::longitudinal_stiffness(Real Fz) const
{
    const auto& c = params.longitudinal;
    const Real dFz = (Fz - c.Fz0) / c.Fz0;
    return c.K0 + c.K_Fz * dFz;
}

Real PacejkaTire::peak_mu_lateral(Real Fz) const
{
    const auto& c = params.lateral;
    const Real dFz = (Fz - c.Fz0) / c.Fz0;
    return c.mu0 + c.mu_Fz * dFz;
}

Real PacejkaTire::peak_mu_longitudinal(Real Fz) const
{
    const auto& c = params.longitudinal;
    const Real dFz = (Fz - c.Fz0) / c.Fz0;
    return c.mu0 + c.mu_Fz * dFz;
}

} // namespace mbd
