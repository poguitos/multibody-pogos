#pragma once

// Optimization infrastructure: Nelder-Mead optimizer + suspension cost functions.
//
// The optimizer is generic and can be used for any scalar objective.
// The suspension cost functions evaluate KinematicSweepResult data.

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/vehicle/suspension/double_wishbone.hpp"
#include "mbd/vehicle/suspension/mcpherson.hpp"
#include "mbd/vehicle/suspension/multilink.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

namespace mbd {

// ============================================================================
// Nelder-Mead optimizer (generic)
// ============================================================================

struct NelderMeadConfig {
    int max_iterations{500};
    Real tol_fun{1e-8};        ///< Convergence: spread of function values
    Real tol_x{1e-8};          ///< Convergence: simplex diameter
    Real initial_step{0.005};  ///< Initial simplex perturbation per dimension
    Real alpha{1.0};           ///< Reflection coefficient
    Real gamma{2.0};           ///< Expansion coefficient
    Real rho{0.5};             ///< Contraction coefficient
    Real sigma{0.5};           ///< Shrink coefficient
};

struct NelderMeadResult {
    VecX best_params;
    Real best_cost{0.0};
    int iterations{0};
    bool converged{false};
    std::vector<Real> cost_history; ///< Best cost at each iteration
};

/// Minimize a scalar function using the Nelder-Mead simplex method.
///
/// \param objective   f(x) -> scalar cost to minimize.
/// \param x0          Initial guess (n-dimensional).
/// \param lower       Lower bounds per dimension (empty = no bounds).
/// \param upper       Upper bounds per dimension (empty = no bounds).
/// \param config      Algorithm parameters.
NelderMeadResult nelder_mead_minimize(
    const std::function<Real(const VecX&)>& objective,
    const VecX& x0,
    const VecX& lower = VecX(),
    const VecX& upper = VecX(),
    const NelderMeadConfig& config = NelderMeadConfig{});

// ============================================================================
// Suspension cost functions
// ============================================================================

/// Individual cost term for suspension optimization.
struct CostTerm {
    enum class Type {
        CamberRange,        ///< Minimize max_camber - min_camber over sweep [rad]
        ToeRange,           ///< Minimize max_toe - min_toe over sweep [rad]
        TargetCamberGain,   ///< (actual_gain - target)^2 [rad/m]
        MaxAbsCamber,       ///< Minimize max(|camber|) over sweep [rad]
        MaxAbsToe           ///< Minimize max(|toe|) over sweep [rad]
    };

    Type type;
    Real weight{1.0};
    Real target{0.0};  ///< For target-based terms

    CostTerm(Type t, Real w = 1.0, Real tgt = 0.0)
        : type(t), weight(w), target(tgt) {}
};

/// Evaluate a cost function from kinematic sweep results.
Real evaluate_suspension_cost(
    const KinematicSweepResult& sweep,
    const std::vector<CostTerm>& terms);

// ============================================================================
// Sweep configuration for optimization
// ============================================================================

struct SweepConfig {
    Real bump_min{-0.03};
    Real bump_max{0.03};
    int n_steps{11};
};

// ============================================================================
// Double-wishbone optimization
// ============================================================================

/// Defines which hardpoint coordinates are free variables for DWB optimization.
struct DwbParameterMapping {
    struct ParamDef {
        enum class Point { LCA_PIVOT, LCA_OUTER, UCA_PIVOT, UCA_OUTER, TIEROD_INNER, TIEROD_OUTER };
        Point point;
        int axis;  ///< 0=X, 1=Y, 2=Z
        Real lower_bound;
        Real upper_bound;
    };

    std::vector<ParamDef> params;

    int dimension() const { return static_cast<int>(params.size()); }

    /// Extract initial parameter vector from a DWB parameter set.
    VecX extract(const DoubleWishboneParams& p) const;

    /// Apply parameter vector to a DWB parameter set.
    DoubleWishboneParams apply(const DoubleWishboneParams& base, const VecX& x) const;

    VecX lower_bounds() const;

    VecX upper_bounds() const;

private:
    static Real get_value(const DoubleWishboneParams& p, const ParamDef& def);

    static void set_value(DoubleWishboneParams& p, const ParamDef& def, Real val);

    static const Vec3& get_point(const DoubleWishboneParams& p, ParamDef::Point pt);

    static Vec3& get_point_mut(DoubleWishboneParams& p, ParamDef::Point pt);
};

/// Result of a DWB optimization.
struct DwbOptimizationResult {
    DoubleWishboneParams optimized_params;
    Real initial_cost{0.0};
    Real final_cost{0.0};
    int iterations{0};
    bool converged{false};
    KinematicSweepResult initial_sweep;
    KinematicSweepResult final_sweep;
};

/// Optimize a double-wishbone suspension's hardpoints.
///
/// \param base_params      Starting hardpoint geometry.
/// \param param_mapping    Which coordinates are free variables + bounds.
/// \param cost_terms       What to optimize (camber range, toe range, etc.).
/// \param sweep_config     Bump travel range and resolution.
/// \param nm_config        Nelder-Mead parameters.
DwbOptimizationResult optimize_dwb(
    const DoubleWishboneParams& base_params,
    const DwbParameterMapping& param_mapping,
    const std::vector<CostTerm>& cost_terms,
    const SweepConfig& sweep_config = SweepConfig{},
    const NelderMeadConfig& nm_config = NelderMeadConfig{});

} // namespace mbd
