#pragma once

// Pacejka Magic Formula tire model (steady-state).
//
// Implements the pure slip Magic Formula for lateral (Fy) and longitudinal (Fx)
// forces, plus combined slip weighting functions.
//
// Reference: Pacejka, "Tire and Vehicle Dynamics", 3rd ed., Chapters 3-4.

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"

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
    static PacejkaTireParams DefaultPassengerCar();
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
                                       const MagicFormulaCoeffs& c);

    /// Evaluate the combined slip weighting function.
    ///   x_other: the "other" slip quantity (alpha for Gx, kappa for Gy)
    ///   c: combined slip coefficients
    /// Returns the weighting factor (1.0 = no reduction).
    static Real evaluate_combined_weight(Real x_other,
                                         const CombinedSlipCoeffs& c);

    /// Compute tire forces from slip quantities and vertical load.
    ///
    /// \param kappa  Slip ratio (positive = traction).
    /// \param alpha  Slip angle [rad] (positive = generates positive Fy).
    /// \param Fz     Vertical load [N] (positive = tire loaded).
    TireForceResult compute(Real kappa, Real alpha, Real Fz) const;

    /// Compute the cornering stiffness Kalpha = dFy/dalpha at alpha=0.
    /// This is the initial slope of the Fy vs alpha curve.
    Real cornering_stiffness(Real Fz) const;

    /// Compute the longitudinal slip stiffness Kkappa = dFx/dkappa at kappa=0.
    Real longitudinal_stiffness(Real Fz) const;

    /// Compute peak lateral friction coefficient at given Fz.
    Real peak_mu_lateral(Real Fz) const;

    /// Compute peak longitudinal friction coefficient at given Fz.
    Real peak_mu_longitudinal(Real Fz) const;
};

} // namespace mbd
