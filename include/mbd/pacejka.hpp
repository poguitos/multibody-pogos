#pragma once

// Pacejka Magic Formula tire model (steady-state).
//
// Implements the pure slip Magic Formula for lateral (Fy) and longitudinal (Fx)
// forces, plus combined slip weighting functions.
//
// Reference: Pacejka, "Tire and Vehicle Dynamics", 3rd ed., Chapters 3-4.

#include "mbd/core.hpp"
#include "mbd/math.hpp"

#include <cmath>
#include <algorithm>

namespace mbd {

// ============================================================================
// Magic Formula coefficients
// ============================================================================

/// Coefficients for one axis of the Magic Formula.
/// Y(x) = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
/// where D, C, B, E are functions of vertical load Fz.
struct MagicFormulaCoeffs {
    // Peak factor: D = mu * Fz (load-dependent friction)
    Real mu0{1.0};       ///< Friction coefficient at Fz0
    Real mu_Fz{0.0};     ///< Friction load sensitivity: mu = mu0 + mu_Fz * dFz

    // Shape factor C (typically 1.3 for Fy, 1.65 for Fx)
    Real C{1.3};

    // Stiffness factor: B = K / (C * D)
    // K (cornering/slip stiffness) = K0 + K_Fz * dFz
    Real K0{50000.0};    ///< Stiffness at nominal load [N/rad] or [N/unit_slip]
    Real K_Fz{0.0};      ///< Stiffness load sensitivity

    // Curvature factor E (controls shape near peak, typically -2 to 1)
    Real E0{-1.0};
    Real E_Fz{0.0};      ///< E = E0 + E_Fz * dFz

    // Nominal load for normalization
    Real Fz0{4000.0};    ///< Nominal vertical load [N]
};

/// Combined slip weighting coefficients.
/// G = D_comb * cos(C_comb * atan(B_comb * x_other))
/// where x_other is the "other" slip quantity.
struct CombinedSlipCoeffs {
    Real B_comb{6.0};
    Real C_comb{1.2};
    Real D_comb{1.0};    ///< Should be 1.0 for no reduction at zero other-slip
};

// ============================================================================
// Full tire parameter set
// ============================================================================

struct PacejkaTireParams {
    MagicFormulaCoeffs lateral;   ///< Fy parameters (slip angle)
    MagicFormulaCoeffs longitudinal; ///< Fx parameters (slip ratio)

    CombinedSlipCoeffs Gx_alpha;  ///< Longitudinal force reduction from slip angle
    CombinedSlipCoeffs Gy_kappa;  ///< Lateral force reduction from slip ratio

    /// Create a reasonable default passenger car tire.
    static PacejkaTireParams DefaultPassengerCar()
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
};

// ============================================================================
// Tire force output
// ============================================================================

struct TireForceResult {
    Real Fx{0.0};        ///< Longitudinal force [N] (positive = traction)
    Real Fy{0.0};        ///< Lateral force [N] (positive = left turn force)
    Real Fz{0.0};        ///< Vertical force [N] (positive = upward, input)
    Real kappa{0.0};     ///< Slip ratio used
    Real alpha{0.0};     ///< Slip angle used [rad]
    Real Fx_pure{0.0};   ///< Pure slip Fx (before combined reduction)
    Real Fy_pure{0.0};   ///< Pure slip Fy (before combined reduction)
    Real Gx{1.0};        ///< Combined slip factor for Fx
    Real Gy{1.0};        ///< Combined slip factor for Fy
};

// ============================================================================
// Pacejka tire model (steady-state)
// ============================================================================

class PacejkaTire {
public:
    PacejkaTireParams params;

    PacejkaTire() : params(PacejkaTireParams::DefaultPassengerCar()) {}
    explicit PacejkaTire(const PacejkaTireParams& p) : params(p) {}

    /// Evaluate the Magic Formula for one axis.
    ///   x: slip input (alpha for lateral, kappa for longitudinal)
    ///   Fz: current vertical load [N]
    ///   c: coefficients for this axis
    /// Returns the force value.
    static Real evaluate_magic_formula(Real x,
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

    /// Evaluate the combined slip weighting function.
    ///   x_other: the "other" slip quantity (alpha for Gx, kappa for Gy)
    ///   c: combined slip coefficients
    /// Returns the weighting factor (1.0 = no reduction).
    static Real evaluate_combined_weight(Real x_other,
                                         const CombinedSlipCoeffs& c)
    {
        const Real Bx = c.B_comb * x_other;
        return c.D_comb * std::cos(c.C_comb * std::atan(Bx));
    }

    /// Compute tire forces from slip quantities and vertical load.
    ///
    /// \param kappa  Slip ratio (positive = traction).
    /// \param alpha  Slip angle [rad] (positive = generates positive Fy).
    /// \param Fz     Vertical load [N] (positive = tire loaded).
    TireForceResult compute(Real kappa, Real alpha, Real Fz) const
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

    /// Compute the cornering stiffness Kalpha = dFy/dalpha at alpha=0.
    /// This is the initial slope of the Fy vs alpha curve.
    Real cornering_stiffness(Real Fz) const
    {
        const auto& c = params.lateral;
        const Real dFz = (Fz - c.Fz0) / c.Fz0;
        return c.K0 + c.K_Fz * dFz;
    }

    /// Compute the longitudinal slip stiffness Kkappa = dFx/dkappa at kappa=0.
    Real longitudinal_stiffness(Real Fz) const
    {
        const auto& c = params.longitudinal;
        const Real dFz = (Fz - c.Fz0) / c.Fz0;
        return c.K0 + c.K_Fz * dFz;
    }

    /// Compute peak lateral friction coefficient at given Fz.
    Real peak_mu_lateral(Real Fz) const
    {
        const auto& c = params.lateral;
        const Real dFz = (Fz - c.Fz0) / c.Fz0;
        return c.mu0 + c.mu_Fz * dFz;
    }

    /// Compute peak longitudinal friction coefficient at given Fz.
    Real peak_mu_longitudinal(Real Fz) const
    {
        const auto& c = params.longitudinal;
        const Real dFz = (Fz - c.Fz0) / c.Fz0;
        return c.mu0 + c.mu_Fz * dFz;
    }
};

} // namespace mbd