#include "mbd/analysis/optimization.hpp"

namespace mbd {

NelderMeadResult nelder_mead_minimize(
    const std::function<Real(const VecX&)>& objective,
    const VecX& x0,
    const VecX& lower,
    const VecX& upper,
    const NelderMeadConfig& config)
{
    const int n = static_cast<int>(x0.size());
    const bool has_bounds = (lower.size() == n && upper.size() == n);

    // Clamp helper
    auto clamp_params = [&](VecX& x) {
        if (!has_bounds) return;
        for (int i = 0; i < n; ++i) {
            x(i) = std::clamp(x(i), lower(i), upper(i));
        }
    };

    // Build initial simplex: n+1 vertices
    std::vector<VecX> simplex(n + 1);
    std::vector<Real> fvals(n + 1);

    simplex[0] = x0;
    clamp_params(simplex[0]);
    fvals[0] = objective(simplex[0]);

    for (int i = 0; i < n; ++i) {
        simplex[i + 1] = x0;
        simplex[i + 1](i) += config.initial_step;
        clamp_params(simplex[i + 1]);
        fvals[i + 1] = objective(simplex[i + 1]);
    }

    NelderMeadResult result;

    for (int iter = 0; iter < config.max_iterations; ++iter) {
        // Sort vertices by function value
        std::vector<int> order(n + 1);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return fvals[a] < fvals[b]; });

        std::vector<VecX> sorted_simplex(n + 1);
        std::vector<Real> sorted_fvals(n + 1);
        for (int i = 0; i <= n; ++i) {
            sorted_simplex[i] = simplex[order[i]];
            sorted_fvals[i]   = fvals[order[i]];
        }
        simplex = sorted_simplex;
        fvals   = sorted_fvals;

        result.cost_history.push_back(fvals[0]);

        // Check convergence: function value spread
        const Real f_spread = fvals[n] - fvals[0];
        if (f_spread < config.tol_fun) {
            result.converged = true;
            break;
        }

        // Check convergence: simplex diameter
        Real max_dist = 0.0;
        for (int i = 1; i <= n; ++i) {
            max_dist = std::max(max_dist, (simplex[i] - simplex[0]).norm());
        }
        if (max_dist < config.tol_x) {
            result.converged = true;
            break;
        }

        // Centroid of all vertices except the worst
        VecX centroid = VecX::Zero(n);
        for (int i = 0; i < n; ++i) {
            centroid += simplex[i];
        }
        centroid /= static_cast<Real>(n);

        // Reflection
        VecX x_r = centroid + config.alpha * (centroid - simplex[n]);
        clamp_params(x_r);
        Real f_r = objective(x_r);

        if (f_r < fvals[0]) {
            // Try expansion
            VecX x_e = centroid + config.gamma * (x_r - centroid);
            clamp_params(x_e);
            Real f_e = objective(x_e);

            if (f_e < f_r) {
                simplex[n] = x_e;
                fvals[n]   = f_e;
            } else {
                simplex[n] = x_r;
                fvals[n]   = f_r;
            }
        } else if (f_r < fvals[n - 1]) {
            // Accept reflection
            simplex[n] = x_r;
            fvals[n]   = f_r;
        } else {
            // Contraction
            VecX x_c;
            Real f_c;

            if (f_r < fvals[n]) {
                // Outside contraction
                x_c = centroid + config.rho * (x_r - centroid);
            } else {
                // Inside contraction
                x_c = centroid + config.rho * (simplex[n] - centroid);
            }
            clamp_params(x_c);
            f_c = objective(x_c);

            if (f_c < std::min(f_r, fvals[n])) {
                simplex[n] = x_c;
                fvals[n]   = f_c;
            } else {
                // Shrink: move all vertices toward the best
                for (int i = 1; i <= n; ++i) {
                    simplex[i] = simplex[0] + config.sigma * (simplex[i] - simplex[0]);
                    clamp_params(simplex[i]);
                    fvals[i] = objective(simplex[i]);
                }
            }
        }

        result.iterations = iter + 1;
    }

    // Find best vertex
    int best = 0;
    for (int i = 1; i <= n; ++i) {
        if (fvals[i] < fvals[best]) best = i;
    }

    result.best_params = simplex[best];
    result.best_cost   = fvals[best];

    return result;
}

Real evaluate_suspension_cost(
    const KinematicSweepResult& sweep,
    const std::vector<CostTerm>& terms)
{
    if (sweep.points.empty()) return std::numeric_limits<Real>::max();

    // Check all points converged
    for (const auto& pt : sweep.points) {
        if (!pt.converged) return std::numeric_limits<Real>::max();
    }

    Real cost = 0.0;

    for (const auto& term : terms) {
        Real value = 0.0;

        switch (term.type) {
            case CostTerm::Type::CamberRange: {
                Real cmin = sweep.points[0].camber;
                Real cmax = sweep.points[0].camber;
                for (const auto& pt : sweep.points) {
                    cmin = std::min(cmin, pt.camber);
                    cmax = std::max(cmax, pt.camber);
                }
                value = cmax - cmin;
                break;
            }
            case CostTerm::Type::ToeRange: {
                Real tmin = sweep.points[0].toe;
                Real tmax = sweep.points[0].toe;
                for (const auto& pt : sweep.points) {
                    tmin = std::min(tmin, pt.toe);
                    tmax = std::max(tmax, pt.toe);
                }
                value = tmax - tmin;
                break;
            }
            case CostTerm::Type::TargetCamberGain: {
                Real gain = sweep.camber_gain();
                Real diff = gain - term.target;
                value = diff * diff;
                break;
            }
            case CostTerm::Type::MaxAbsCamber: {
                for (const auto& pt : sweep.points) {
                    value = std::max(value, std::abs(pt.camber));
                }
                break;
            }
            case CostTerm::Type::MaxAbsToe: {
                for (const auto& pt : sweep.points) {
                    value = std::max(value, std::abs(pt.toe));
                }
                break;
            }
        }

        cost += term.weight * value;
    }

    return cost;
}

VecX DwbParameterMapping::extract(const DoubleWishboneParams& p) const
{
    VecX x(dimension());
    for (int i = 0; i < dimension(); ++i) {
        x(i) = get_value(p, params[i]);
    }
    return x;
}

DoubleWishboneParams DwbParameterMapping::apply(const DoubleWishboneParams& base, const VecX& x) const
{
    DoubleWishboneParams p = base;
    for (int i = 0; i < dimension(); ++i) {
        set_value(p, params[i], x(i));
    }
    return p;
}

VecX DwbParameterMapping::lower_bounds() const
{
    VecX lb(dimension());
    for (int i = 0; i < dimension(); ++i) lb(i) = params[i].lower_bound;
    return lb;
}

VecX DwbParameterMapping::upper_bounds() const
{
    VecX ub(dimension());
    for (int i = 0; i < dimension(); ++i) ub(i) = params[i].upper_bound;
    return ub;
}

Real DwbParameterMapping::get_value(const DoubleWishboneParams& p, const ParamDef& def)
{
    const Vec3& pt = get_point(p, def.point);
    return pt(def.axis);
}

void DwbParameterMapping::set_value(DoubleWishboneParams& p, const ParamDef& def, Real val)
{
    Vec3& pt = get_point_mut(p, def.point);
    pt(def.axis) = val;
}

const Vec3& DwbParameterMapping::get_point(const DoubleWishboneParams& p, ParamDef::Point pt)
{
    switch (pt) {
        case ParamDef::Point::LCA_PIVOT:    return p.lca_pivot;
        case ParamDef::Point::LCA_OUTER:    return p.lca_outer;
        case ParamDef::Point::UCA_PIVOT:    return p.uca_pivot;
        case ParamDef::Point::UCA_OUTER:    return p.uca_outer;
        case ParamDef::Point::TIEROD_INNER: return p.tierod_inner;
        case ParamDef::Point::TIEROD_OUTER: return p.tierod_outer;
    }
    return p.lca_pivot; // unreachable
}

Vec3& DwbParameterMapping::get_point_mut(DoubleWishboneParams& p, ParamDef::Point pt)
{
    switch (pt) {
        case ParamDef::Point::LCA_PIVOT:    return p.lca_pivot;
        case ParamDef::Point::LCA_OUTER:    return p.lca_outer;
        case ParamDef::Point::UCA_PIVOT:    return p.uca_pivot;
        case ParamDef::Point::UCA_OUTER:    return p.uca_outer;
        case ParamDef::Point::TIEROD_INNER: return p.tierod_inner;
        case ParamDef::Point::TIEROD_OUTER: return p.tierod_outer;
    }
    return p.lca_pivot; // unreachable
}

DwbOptimizationResult optimize_dwb(
    const DoubleWishboneParams& base_params,
    const DwbParameterMapping& param_mapping,
    const std::vector<CostTerm>& cost_terms,
    const SweepConfig& sweep_config,
    const NelderMeadConfig& nm_config)
{
    DwbOptimizationResult result;
    result.optimized_params = base_params;

    // A bump sweep of the kinematic corner built from p, on the kernel.
    auto sweep_of = [&sweep_config](const DoubleWishboneParams& p) {
        kernel::System sys;
        const auto dwb = build_double_wishbone_corner(sys, p);
        Kinematics k(sys);
        return sweep_bump_travel(k, dwb.upright_body, sweep_config.bump_min,
                                 sweep_config.bump_max, sweep_config.n_steps);
    };

    // Objective function
    auto objective = [&](const VecX& x) -> Real {
        return evaluate_suspension_cost(sweep_of(param_mapping.apply(base_params, x)), cost_terms);
    };

    // Initial cost and sweep
    result.initial_sweep = sweep_of(base_params);
    result.initial_cost = evaluate_suspension_cost(result.initial_sweep, cost_terms);

    // Run optimizer
    VecX x0 = param_mapping.extract(base_params);
    auto nm_result = nelder_mead_minimize(
        objective, x0,
        param_mapping.lower_bounds(),
        param_mapping.upper_bounds(),
        nm_config);

    result.optimized_params = param_mapping.apply(base_params, nm_result.best_params);
    result.final_cost = nm_result.best_cost;
    result.iterations = nm_result.iterations;
    result.converged  = nm_result.converged;

    // Final sweep with optimized params
    result.final_sweep = sweep_of(result.optimized_params);

    return result;
}

} // namespace mbd
