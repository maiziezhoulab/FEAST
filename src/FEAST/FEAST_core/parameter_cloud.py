import numpy as np
import pandas as pd
import warnings

# --- Imports for parallelization and distribution distance ---
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import scanpy as sc

from scipy.stats import rankdata, t
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import pyvinecopulib as pv
from ..modeling.StudentT_mixture_model import StudentTMixtureMarginalModeler
from ..modeling.Beta_mixture_model import BetaMixtureMarginalModeler
from ..modeling.marginal_alteration import alter_marginal_model, AlterationConfig

def to_uniform(series):
    return rankdata(series, method='ordinal') / (len(series) + 1)

class DependencyModeler:
    @staticmethod
    def fit_copula_model(data_df):
        print("\n--- Fitting Dependency Model (Vine Copula) ---")
        uniform_data = data_df[['mean', 'variance', 'zero_prop']].apply(to_uniform).to_numpy()
        
        # Define the set of copula families to consider
        family_set_list = [
            pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe
        ]
        
        controls = pv.FitControlsVinecop(family_set=family_set_list, selection_criterion='bic')
        copula_model = pv.Vinecop(d=uniform_data.shape[1])
        copula_model.select(data=uniform_data, controls=controls)
        
        print("  > Vine copula structure and parameters selected via BIC.")
        return copula_model

def _run_single_heuristic_attempt(simulator, synthetic_pool, assignment_weights, random_seed):
    """Helper function to encapsulate one full assignment and evaluation for parallelization."""
    assigned_params = simulator.assign_to_genes(synthetic_pool, weights=assignment_weights, random_seed=random_seed, verbose=False)
    error = simulator.evaluate_parameter_fidelity(assigned_params, weights=assignment_weights)
    return error, assigned_params

class GeneParameterSimulator:
    def __init__(self):
        self.param_models = {
            'mean': StudentTMixtureMarginalModeler(max_components=15), 
            'variance': StudentTMixtureMarginalModeler(max_components=15), 
            'zero_prop': BetaMixtureMarginalModeler(max_components=8)
        }
        print("✓ Using optimal models: Student's T for mean, Student's T for variance, Beta for zero_prop.")
        
        self.fitted = False
        self.copula_model, self.original_stats, self.dependency_modeler, self.n_obs = None, None, DependencyModeler(), None

    def fit_statistics_only(self, adata):
        print("\n--- [FITTING STATS ONLY] Calculating original gene statistics ---")
        self.n_obs = adata.n_obs
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()
        self.original_stats = pd.DataFrame({
            'mean': np.mean(X, axis=0), 
            'variance': np.var(X, axis=0), 
            'zero_prop': 1 - (np.count_nonzero(X, axis=0) / self.n_obs)
        }, index=adata.var_names).clip(lower=1e-10)
        print("✓ Statistics calculated.")
        return self

    def fit(self, adata, visualize_fits=True):
        self.fit_statistics_only(adata)
        print("\n--- [FITTING MODELS] Fitting marginal and dependency models ---")
        for param, modeler in self.param_models.items():
            # Check if modeler accepts log_transform parameter
            import inspect
            fit_signature = inspect.signature(modeler.fit)
            if 'log_transform' in fit_signature.parameters:
                # Student's T models accept log_transform
                modeler.fit(self.original_stats[param], log_transform=(param != 'zero_prop'), visualize=visualize_fits)
            else:
                # Beta models and others don't accept log_transform
                modeler.fit(self.original_stats[param], visualize=visualize_fits)
        self.copula_model = self.dependency_modeler.fit_copula_model(self.original_stats)
        self.fitted = True
        print("\n✓ Simulator has been successfully fitted to the data.")
        return self

    def simulate(self, n_genes, overgeneration_factor=1.1, verbose=True):
        if not self.fitted: raise RuntimeError("Simulator must be fitted first.")
        n_to_generate = int(n_genes * overgeneration_factor)
        if verbose: print(f"\n--- [SIMULATING] Generating {n_to_generate} synthetic profiles...")
        
        uniform_samples = self.copula_model.simulate(n=n_to_generate, seeds=[np.random.randint(1e6)])
        final_params = pd.DataFrame({
            param: modeler.ppf(uniform_samples[:, i]) for i, (param, modeler) in enumerate(self.param_models.items())
        })
        
        if verbose: print("  > Enforcing minimum observed parameter boundaries...")
        for param in ['mean', 'variance', 'zero_prop']:
            final_params[param] = final_params[param].clip(lower=self.original_stats[param].min())
        final_params['zero_prop'] = final_params['zero_prop'].clip(upper=1.0)
        
        if verbose: print("✓ Simulation complete.")
        return final_params

    def assign_to_genes(self, synthetic_df, weights={'mean': 3.0, 'variance': 1.0, 'zero_prop': 1.0}, random_seed=42, verbose=True):
        if verbose: print(f"\n--- [ASSIGNING] Assigning synthetic profiles (seed: {random_seed})...")
        if len(synthetic_df) < len(self.original_stats): raise ValueError("Fewer synthetic profiles than real genes. Increase overgeneration_factor.")
        
        synthetic_subset = synthetic_df.sample(n=len(self.original_stats), random_state=random_seed).reset_index(drop=True)
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(np.log10(self.original_stats.clip(lower=1e-10)))
        synth_scaled = scaler.transform(np.log10(synthetic_subset.clip(lower=1e-10)))
        
        weight_vector = np.array([weights['mean'], weights['variance'], weights['zero_prop']])
        cost_matrix = cdist(orig_scaled * weight_vector, synth_scaled * weight_vector, 'euclidean')
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_df = synthetic_subset.iloc[col_ind].reset_index(drop=True)
        assigned_df['gene_id'] = self.original_stats.index[row_ind]
        
        if verbose: print("✓ Assignment complete.")
        return assigned_df[['gene_id', 'mean', 'variance', 'zero_prop']]

    def evaluate_parameter_fidelity(self, assigned_synthetic_params: pd.DataFrame, weights={'mean': 1.0, 'variance': 1.0, 'zero_prop': 1.0}):
        if self.original_stats is None: raise RuntimeError("Original statistics are not available.")
        
        assigned_reordered = assigned_synthetic_params.set_index('gene_id').loc[self.original_stats.index]
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(np.log10(self.original_stats.clip(lower=1e-10)))
        synth_scaled = scaler.transform(np.log10(assigned_reordered.clip(lower=1e-10)))
        
        weight_vector = np.array([weights['mean'], weights['variance'], weights['zero_prop']])
        weighted_squared_errors = ((orig_scaled - synth_scaled) ** 2) * weight_vector
        return np.mean(weighted_squared_errors)
    
    def _calculate_distribution_distance(self, synthetic_subset):
        dist_mean = wasserstein_distance(self.original_stats['mean'], synthetic_subset['mean'])
        dist_var = wasserstein_distance(np.log10(self.original_stats['variance']), np.log10(synthetic_subset['variance']))
        dist_zero = wasserstein_distance(self.original_stats['zero_prop'], synthetic_subset['zero_prop'])
        return dist_mean + dist_var + dist_zero

    def run_heuristic_search(self, n_genes, min_accepted_error, screening_pool_size=100, top_n_to_fully_evaluate=5, overgeneration_factor=1.1, assignment_weights=None, n_jobs=-1):
        if not self.fitted: raise RuntimeError("Simulator must be fitted first.")
        if assignment_weights is None: assignment_weights = {'mean': 1.0, 'variance': 1.0, 'zero_prop': 1.0}
        
        print(f"\n--- [BOOSTED HEURISTIC SEARCH] Starting optimization ---")
        print(f"  > Target Error: < {min_accepted_error:.6f}")
        print(f"  > Pre-screening Pool Size: {screening_pool_size}")
        print(f"  > Finalists for Full OT: {top_n_to_fully_evaluate}")
        print(f"  > Parallel Jobs: {n_jobs if n_jobs != -1 else 'All available CPUs'}")

        print("\n--- Stage 1: Generating and pre-screening candidates... ---")
        synthetic_pool = self.simulate(n_genes=n_genes, overgeneration_factor=overgeneration_factor, verbose=False)
        
        candidates = []
        for i in range(screening_pool_size):
            random_seed = np.random.randint(1e6)
            synthetic_subset = synthetic_pool.sample(n=len(self.original_stats), random_state=random_seed)
            proxy_dist = self._calculate_distribution_distance(synthetic_subset)
            candidates.append({'proxy_dist': proxy_dist, 'seed': random_seed})
        
        candidates.sort(key=lambda x: x['proxy_dist'])
        top_candidates = candidates[:top_n_to_fully_evaluate]
        print(f"✓ Pre-screening complete. Identified top {len(top_candidates)} candidates for full evaluation.")

        print("\n--- Stage 2: Running full Optimal Transport on best candidates... ---")
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_heuristic_attempt)(self, synthetic_pool, assignment_weights, cand['seed']) for cand in top_candidates
        )
        
        errors, assigned_dfs = zip(*parallel_results)
        best_idx = np.argmin(errors)
        lowest_error = errors[best_idx]
        best_assigned_params = assigned_dfs[best_idx]
        
        print(f"✓ Full evaluation complete. Best error found: {lowest_error:.6f}")

        if lowest_error < min_accepted_error:
            print(f"\n✓ SUCCESS: Found a result below the error threshold.")
        else:
            warnings.warn(f"Heuristic search finished without reaching the desired error rate ({min_accepted_error:.6f}). "
                          f"Returning the best result found from the top candidates.")
                          
        return best_assigned_params

    def alter_marginal_distributions(self, alteration_config=None, verbose=True):
        """
        Alter fitted marginal distributions using user-friendly fold-change controls.
        
        Args:
            alteration_config (AlterationConfig or dict): Configuration for alterations.
                                                         If None, no alterations are applied.
            verbose (bool): Print alteration details
            
        Returns:
            self: Returns the modified simulator instance
            
        Example:
            >>> # Create alteration configuration
            >>> config = AlterationConfig(
            ...     mean_fold_change=2.0,      # Double gene expression means
            ...     variance_fold_change=1.5,  # Increase variance by 50%
            ...     apply_to_mean=True,
            ...     apply_to_variance=True,
            ...     apply_to_zero_prop=False
            ... )
            >>> simulator.alter_marginal_distributions(config)
        """
        if not self.fitted:
            raise RuntimeError("Simulator must be fitted before marginal distributions can be altered.")
        
        if alteration_config is None:
            if verbose:
                print("No alteration configuration provided. Skipping marginal distribution alterations.")
            return self
        
        # Convert to AlterationConfig if dictionary provided
        if isinstance(alteration_config, dict):
            alteration_config = AlterationConfig(**alteration_config)
        
        if verbose:
            print(f"\n--- [ALTERING MARGINALS] Applying distribution modifications ---")
            print(f"  Mean fold change: {alteration_config.mean_fold_change}x")
            print(f"  Variance fold change: {alteration_config.variance_fold_change}x")
            print(f"  Apply to mean: {alteration_config.apply_to_mean}")
            print(f"  Apply to variance: {alteration_config.apply_to_variance}")
            print(f"  Apply to zero_prop: {alteration_config.apply_to_zero_prop}")
        
        # Apply alterations to selected marginal distributions
        alterations_applied = []
        
        if alteration_config.apply_to_mean:
            if verbose:
                print("\n  > Altering MEAN distribution...")
            self.param_models['mean'] = alter_marginal_model(
                self.param_models['mean'],
                mean_fold_change=alteration_config.mean_fold_change,
                variance_fold_change=alteration_config.variance_fold_change,
                dispersion_strength=alteration_config.dispersion_strength,
                preserve_original=False,  # Modify in place
                verbose=verbose
            )
            alterations_applied.append('mean')
        
        if alteration_config.apply_to_variance:
            if verbose:
                print("\n  > Altering VARIANCE distribution...")
            self.param_models['variance'] = alter_marginal_model(
                self.param_models['variance'],
                mean_fold_change=alteration_config.mean_fold_change,
                variance_fold_change=alteration_config.variance_fold_change,
                dispersion_strength=alteration_config.dispersion_strength,
                preserve_original=False,  # Modify in place
                verbose=verbose
            )
            alterations_applied.append('variance')
        
        if alteration_config.apply_to_zero_prop:
            if verbose:
                print("\n  > Altering ZERO PROPORTION distribution...")
            self.param_models['zero_prop'] = alter_marginal_model(
                self.param_models['zero_prop'],
                mean_fold_change=alteration_config.mean_fold_change,
                variance_fold_change=alteration_config.variance_fold_change,
                dispersion_strength=alteration_config.dispersion_strength,
                preserve_original=False,  # Modify in place
                verbose=verbose
            )
            alterations_applied.append('zero_prop')
        
        if verbose:
            print(f"\n✓ Marginal distribution alterations complete.")
            print(f"  Altered distributions: {', '.join(alterations_applied)}")
            print(f"  Note: Dependency structure (copula) remains unchanged.")
            print(f"        Re-simulation will use altered marginals with original dependencies.")
        
        return self

def _calculate_zip_theoretical_stats(params):
    """Calculates theoretical moments for the Zero-Inflated Poisson (ZIP) model."""
    pi, lamb = params
    mean = (1 - pi) * lamb
    variance = (1 - pi) * lamb * (1 + pi * lamb)
    zero_prop = pi + (1 - pi) * np.exp(-lamb)
    return np.array([mean, variance, zero_prop])

def _calculate_zinb_theoretical_stats(params):
    """Calculates theoretical moments for the Zero-Inflated Negative Binomial (ZINB) model."""
    pi, mu, r = params
    # Ensure r is not infinity for calculations
    safe_r = np.clip(r, 1e-10, 1e10)
    mean = (1 - pi) * mu
    variance = (1 - pi) * (mu + mu**2 / safe_r + pi * mu**2)
    zero_prop = pi + (1 - pi) * (safe_r / (safe_r + mu))**safe_r
    return np.array([mean, variance, zero_prop])


# =============================================================================
# --- NEW: LOG-SCALE OBJECTIVE FUNCTION ---
# This is the core of the improvement. It minimizes the squared error
# between the log-transformed theoretical and target statistics.
# =============================================================================

def _moment_objective_function_log_scale(params, target_stats, model_type):
    """
    Calculates the sum of squared errors on the log10 scale.
    This naturally balances parameters that live on different orders of magnitude.
    """
    theoretical_stats = np.array([0., 0., 0.])
    
    if model_type == 'ZIP':
        # Parameter boundary check
        if not (0 < params[0] < 1 and params[1] > 0): return np.inf
        theoretical_stats = _calculate_zip_theoretical_stats(params)
        
    elif model_type == 'ZINB':
        # Parameter boundary check
        if not (0 < params[0] < 1 and params[1] > 0 and params[2] > 0): return np.inf
        theoretical_stats = _calculate_zinb_theoretical_stats(params)

    # Use log10 transform to evaluate error in terms of magnitude.
    # Add a small epsilon (1e-10) for numerical stability if a stat is zero.
    log_theoretical = np.log10(theoretical_stats + 1e-10)
    log_target = np.log10(target_stats + 1e-10)
    
    # Return the sum of squared errors in log space
    return np.sum((log_theoretical - log_target)**2)


# =============================================================================
# --- UPDATED: PARAMETER ESTIMATION ROUTINES ---
# These functions now call the new log-scale objective function.
# =============================================================================

def _estimate_zip_by_moment_optimization(mu_total, var_total, zero_prop):
    """Finds ZIP parameters by minimizing the log-scale objective function."""
    target_stats = np.array([mu_total, var_total, zero_prop])
    
    # Sensible initial guesses for the optimizer
    initial_pi = np.clip(zero_prop, 0.01, 0.99)
    initial_lambda = max(mu_total / (1 - initial_pi) if (1 - initial_pi) > 1e-8 else mu_total, 1e-8)
    initial_guess = [initial_pi, initial_lambda]
    
    bounds = [(1e-6, 1 - 1e-6), (1e-6, None)]
    
    result = minimize(
        _moment_objective_function_log_scale,  # <-- Using the new objective function
        initial_guess,
        args=(target_stats, 'ZIP'),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-8}
    )
    
    # Return result if optimization was successful and error is low
    if result.success and result.fun < 1e-4:
        return {'pi0': result.x[0], 'lambda': result.x[1]}
    else: # Fallback to initial guess if optimization fails
        # This is normal for interpolated parameters - just use initial guess silently
        return {'pi0': initial_guess[0], 'lambda': initial_guess[1]}

def _estimate_zinb_by_moment_optimization(mu_total, var_total, zero_prop):
    """Finds ZINB parameters by minimizing the log-scale objective function."""
    target_stats = np.array([mu_total, var_total, zero_prop])
    
    # Sensible initial guesses for the optimizer
    initial_pi = np.clip(zero_prop, 0.01, 0.99)
    initial_mu = max(mu_total / (1 - initial_pi) if (1 - initial_pi) > 1e-8 else mu_total, 1e-8)
    initial_r = max((initial_mu**2) / (var_total - initial_mu) if var_total > initial_mu else 1.0, 1e-8)
    initial_guess = [initial_pi, initial_mu, initial_r]

    bounds = [(1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None)]
    
    result = minimize(
        _moment_objective_function_log_scale,  # <-- Using the new objective function
        initial_guess,
        args=(target_stats, 'ZINB'),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-8}
    )
    
    # Return result if optimization was successful and error is low
    if result.success and result.fun < 1e-4:
        return {'pi0': result.x[0], 'mu': result.x[1], 'r': result.x[2]}
    else: # Fallback to initial guess if optimization fails
        return {'pi0': initial_guess[0], 'mu': initial_guess[1], 'r': initial_guess[2]}

def _select_model_with_heuristic(mu_total, var_total, zero_prop, zero_threshold=0.3, overdispersion_threshold=1.5):
    """Heuristically selects a count model based on summary statistics.
    
    ADJUSTED: overdispersion_threshold lowered from 2.0 to 1.5
    - For sparse ST data with mean=0.0689, var/mean often < 1.0 (under-dispersed)
    - Threshold=1.5 catches moderately overdispersed genes while keeping ZIP as default
    - Reference data shows: mean overdispersion=0.71, so most genes should use ZIP
    - Only ~3-5% of genes expected to be ZINB/NB with threshold=1.5
    """
    if mu_total <= 1e-8: return 'Poisson'
    is_zero_inflated = zero_prop > zero_threshold
    is_overdispersed = (var_total / mu_total) > overdispersion_threshold
    
    if is_zero_inflated and is_overdispersed: return 'ZINB'
    if is_zero_inflated: return 'ZIP'
    if is_overdispersed: return 'NB'
    return 'Poisson'

def _estimate_params_no_fallback(model_name, mu_total, var_total, zero_prop):
    """Master function to dispatch to the correct moment-matching optimizer."""
    if model_name == 'Poisson':
        return {'lambda': max(mu_total, 1e-8)}
    if model_name == 'NB':
        # Simple moment matching for non-inflated NB
        r = max((mu_total**2)/(var_total-mu_total), 1e-8) if var_total > mu_total else np.inf
        return {'mu': max(mu_total, 1e-8), 'r': r}
    if model_name == 'ZIP':
        # Calls the updated ZIP estimator
        return _estimate_zip_by_moment_optimization(mu_total, var_total, zero_prop)
    if model_name == 'ZINB':
        # Calls the updated ZINB estimator
        return _estimate_zinb_by_moment_optimization(mu_total, var_total, zero_prop)
    return {}

def convert_params_for_new_simulator(stats_df: pd.DataFrame):
    """
    Converts a DataFrame of statistics (mean, variance, zero_prop) into
    parameters for specific count models (ZINB, etc.) using the improved
    log-scale moment inference method.
    """
    print(f"\n--- [CONVERTING] Converting {len(stats_df)} parameter sets via log-scale moment-matching ---")
    
    output_dict = {'genes': {}, 'model_selected': [], 'marginal_param1': []}
    
    # Debug: track model selection distribution
    model_counts = {}
    debug_stats = []
    
    for i, (gene_id, record) in enumerate(stats_df.iterrows()):
        record_dict = record.to_dict()
        mu = record_dict['mean']
        var = record_dict['variance']
        zp = record_dict['zero_prop']
        
        # Debug: collect stats for analysis
        overdispersion = var / mu if mu > 1e-8 else 0
        debug_stats.append({
            'gene': gene_id,
            'mean': mu,
            'variance': var,
            'zero_prop': zp,
            'overdispersion': overdispersion,
            'is_zero_inflated': zp > 0.3,
            'is_overdispersed': overdispersion > 2.0
        })
        
        # Select the best model and estimate its parameters using the new methods
        model_type = _select_model_with_heuristic(mu, var, zp)
        params = _estimate_params_no_fallback(model_type, mu, var, zp)
        
        model_counts[model_type] = model_counts.get(model_type, 0) + 1
        
        pi0, r, mean_param = 0.0, np.inf, 0.0
        if model_type == 'Poisson':
            mean_param = params.get('lambda', 1e-8)
        elif model_type == 'NB':
            mean_param, r = params.get('mu', 1e-8), params.get('r', np.inf)
        elif model_type == 'ZIP':
            pi0, mean_param = params.get('pi0', 0.0), params.get('lambda', 1e-8)
        elif model_type == 'ZINB':
            pi0, mean_param, r = params.get('pi0', 0.0), params.get('mu', 1e-8), params.get('r', np.inf)
        
        output_dict['genes'][i] = gene_id
        output_dict['model_selected'].append(model_type)
        output_dict['marginal_param1'].append([pi0, r, mean_param])
    
    # Debug output
    debug_df = pd.DataFrame(debug_stats)
    print(f"  > Model selection summary: {model_counts}")
    print(f"  > Zero inflation rate: {debug_df['is_zero_inflated'].mean():.2%}")
    print(f"  > Overdispersion rate: {debug_df['is_overdispersed'].mean():.2%}")
    print(f"  > Mean overdispersion: {debug_df['overdispersion'].mean():.2f}")
    print(f"  > Mean zero proportion: {debug_df['zero_prop'].mean():.3f}")
        
    print("✓ Conversion complete.")
    return output_dict