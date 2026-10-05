#include "mbd/forces/pacejka.hpp"

namespace mbd {

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
                                         const MagicFormulaCoeffs& c)
{
    const Real dFz = (Fz - c.Fz0) / c.Fz0;

    // Peak factor
    const Real mu = c.mu0 + c.mu_Fz * dFz;
    const Real D  = mu * Fz;

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

TireForceResult PacejkaTire::compute(Real kappa, Real alpha, Real Fz) const
{
    TireForceResult r;
    r.kappa = kappa;
    r.alpha = alpha;
    r.Fz    = Fz;

    if (Fz <= Real(0.0)) {
        return r; // No contact
    }

    // Pure slip forces
    r.Fx_pure = evaluate_magic_formula(kappa, Fz, params.longitudinal);
    r.Fy_pure = evaluate_magic_formula(alpha, Fz, params.lateral);

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
