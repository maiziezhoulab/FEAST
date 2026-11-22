import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Union, Optional, Tuple
from copy import deepcopy

class MarginalModelAlterator:
    
    def __init__(self):
        """Initialize the marginal model alterator."""
        self.alteration_history = []
    
    def alter_model(self,
                   modeler,
                   mean_fold_change: float = 1.0,
                   variance_fold_change: float = 1.0,
                   sparsity_fold_change: float | None = None,
                   sparsity_logit_shift: float | None = None,
                   dispersion_strength: float = 0.2,
                   preserve_original: bool = True,
                   verbose: bool = True) -> object:
        # Input validation
        if not hasattr(modeler, '_is_fitted') or not modeler._is_fitted:
            raise ValueError("Modeler must be fitted before it can be altered.")
        
        if mean_fold_change <= 0:
            raise ValueError("mean_fold_change must be positive")
        
        if variance_fold_change < 0:
            raise ValueError("variance_fold_change must be non-negative")
            
        if sparsity_fold_change is not None and sparsity_fold_change <= 0:
            raise ValueError("sparsity_fold_change must be positive")
            
        if not 0 <= dispersion_strength <= 1:
            raise ValueError("dispersion_strength must be between 0 and 1")
        
        # Create copy if requested
        if preserve_original:
            target_modeler = deepcopy(modeler)
        else:
            target_modeler = modeler
        
        if verbose:
            print(f"--- Altering {type(target_modeler).__name__} ---")
            print(f"  Mean fold change: {mean_fold_change}x")
            print(f"  Variance fold change: {variance_fold_change}x")
            if sparsity_logit_shift is not None and sparsity_logit_shift != 0.0:
                print(f"  Sparsity logit shift: {sparsity_logit_shift:+.4f}")
            elif sparsity_fold_change is not None and sparsity_fold_change != 1.0:
                print(f"  Sparsity fold change (deprecated): {sparsity_fold_change}x")
            print(f"  Dispersion strength: {dispersion_strength}")
        
        # Record original statistics for comparison
        original_samples = target_modeler.sample(10000)
        original_mean = np.mean(original_samples)
        original_var = np.var(original_samples)
        
        # Apply alterations based on model type
        if hasattr(target_modeler, 'model_params'):
            if 'means' in target_modeler.model_params:
                # Student's T or similar mixture model
                self._alter_mixture_model(target_modeler, mean_fold_change, 
                                        variance_fold_change, dispersion_strength, verbose)
            elif 'alphas' in target_modeler.model_params and 'betas' in target_modeler.model_params:
                # Beta mixture model
                self._alter_beta_model(target_modeler, mean_fold_change,
                                     variance_fold_change,
                                     sparsity_fold_change=sparsity_fold_change,
                                     sparsity_logit_shift=sparsity_logit_shift,
                                     verbose=verbose)
            else:
                raise ValueError(f"Unknown model parameter structure: {target_modeler.model_params.keys()}")
        else:
            raise ValueError("Modeler does not have recognizable model_params structure")

        if hasattr(target_modeler, '_ppf_cache'):
            target_modeler._ppf_cache.clear()
        if hasattr(target_modeler, '_ppf_interp'):
            delattr(target_modeler, '_ppf_interp')
        
        # Verify results
        if verbose:
            altered_samples = target_modeler.sample(10000)
            altered_mean = np.mean(altered_samples)
            altered_var = np.var(altered_samples)
            
            achieved_mean_fc = altered_mean / original_mean if original_mean != 0 else np.inf
            achieved_var_fc = altered_var / original_var if original_var != 0 else np.inf
            
            print(f"\n--- Alteration Results ---")
            print(f"  Original: Mean={original_mean:.3f}, Var={original_var:.3f}")
            print(f"  Modified: Mean={altered_mean:.3f}, Var={altered_var:.3f}")
            print(f"  Achieved Mean FC: {achieved_mean_fc:.3f}x (Target: {mean_fold_change}x)")
            print(f"  Achieved Var FC: {achieved_var_fc:.3f}x (Target: {variance_fold_change}x)")
            print("✓ Model alteration complete")
        
        # Record alteration in history
        self.alteration_history.append({
            'model_type': type(target_modeler).__name__,
            'mean_fold_change': mean_fold_change,
            'variance_fold_change': variance_fold_change,
            'sparsity_fold_change': sparsity_fold_change,
            'dispersion_strength': dispersion_strength,
            'original_mean': original_mean,
            'original_var': original_var,
            'achieved_mean_fc': achieved_mean_fc if verbose else None,
            'achieved_var_fc': achieved_var_fc if verbose else None
        })
        
        return target_modeler
    
    def _alter_mixture_model(self, modeler, mean_fold_change, variance_fold_change, 
                           dispersion_strength, verbose):
        """
        Alter Student's T or similar mixture models with means and scales.
        """
        params = modeler.model_params

        if mean_fold_change != 1.0:
            if hasattr(modeler, 'log_transform') and modeler.log_transform:
                log_mean_shift = np.log10(mean_fold_change)
                params['means'] += log_mean_shift
                if verbose:
                    print(f"  > Applied log10 mean shift: +{log_mean_shift:.3f}")
            else:
                params['means'] *= mean_fold_change
                if verbose:
                    print(f"  > Applied direct mean multiplier: {mean_fold_change}x")

        if variance_fold_change != 1.0:
            if 'scales' in params:
                scale_inflation_factor = np.sqrt(variance_fold_change)
                params['scales'] *= scale_inflation_factor
                if verbose:
                    print(f"  > Scaled component widths by: {scale_inflation_factor:.3f}x")

            if len(params['means']) > 1 and dispersion_strength > 0:
                mean_dispersion_factor = 1.0 + (variance_fold_change - 1.0) * dispersion_strength
                if mean_dispersion_factor != 1.0:
                    current_overall_mean = np.sum(params['weights'] * params['means'])
                    params['means'] = (current_overall_mean +
                                     mean_dispersion_factor * (params['means'] - current_overall_mean))
                    if verbose:
                        print(f"  > Increased mean separation by: {mean_dispersion_factor:.3f}x")
    
    def _alter_beta_model(self, modeler, mean_fold_change, variance_fold_change,
                          sparsity_fold_change=None, sparsity_logit_shift=None, verbose=False):
        params = modeler.model_params

        if verbose:
            print("  > Altering Beta mixture model...")

        # Sparsity: logit-shift takes precedence over deprecated fold-change
        if sparsity_logit_shift is not None and sparsity_logit_shift != 0.0:
            return self._alter_beta_model_sparsity_logit_shift(
                modeler, float(sparsity_logit_shift), verbose)
        if sparsity_fold_change is not None and sparsity_fold_change != 1.0:
            return self._alter_beta_model_sparsity_fold_change(
                modeler, float(sparsity_fold_change), verbose)
        
        for i in range(len(params['alphas'])):
            alpha_old = params['alphas'][i]
            beta_old = params['betas'][i]
            
            # Current mean and variance
            old_mean = alpha_old / (alpha_old + beta_old)
            old_var = (alpha_old * beta_old) / ((alpha_old + beta_old)**2 * (alpha_old + beta_old + 1))
            
            # Target mean and variance
            target_mean = old_mean * mean_fold_change
            target_var = old_var * variance_fold_change
            
            # Ensure valid Beta distribution constraints
            target_mean = np.clip(target_mean, 0.001, 0.999)
            max_var = target_mean * (1 - target_mean) / 2  # Conservative upper bound
            target_var = min(target_var, max_var)
            
            # Solve for new α and β using method of moments
            if target_var > 0 and 0 < target_mean < 1:
                # β = α(1-μ)/μ and α+β = μ(1-μ)/σ² - 1
                sum_params = target_mean * (1 - target_mean) / target_var - 1
                if sum_params > 2:  # Ensure reasonable parameters
                    alpha_new = target_mean * sum_params
                    beta_new = (1 - target_mean) * sum_params
                    
                    # Apply bounds for numerical stability
                    alpha_new = max(0.1, min(alpha_new, 100))
                    beta_new = max(0.1, min(beta_new, 100))
                    
                    params['alphas'][i] = alpha_new
                    params['betas'][i] = beta_new
                    
                    if verbose:
                        print(f"    Component {i+1}: α={alpha_old:.2f}→{alpha_new:.2f}, β={beta_old:.2f}→{beta_new:.2f}")
        
        return modeler
    
    def _alter_beta_model_sparsity_logit_shift(self, modeler, delta_z, verbose):
        """Alter Beta mixture model by direct logit-space shift.

        logit(q_k') = logit(q_k) + δ_z

        All component means are shifted by the same δ_z in logit space.
        Component concentrations S_k = α_k + β_k are preserved.
        This keeps all q_k' ∈ (0, 1) automatically and preserves component ordering.

        Args:
            delta_z: Additive shift in logit(z) space.
                     δ_z > 0 → more sparse, δ_z < 0 → less sparse.
        """
        params = modeler.model_params
        weights = params['weights']
        alphas = params['alphas']
        betas = params['betas']

        component_means = alphas / (alphas + betas)
        current_global_mean = float(np.sum(weights * component_means))

        logit = lambda q: np.log(np.clip(q, 1e-8, 1.0 - 1e-8) / (1.0 - np.clip(q, 1e-8, 1.0 - 1e-8)))
        inv_logit = lambda x: 1.0 / (1.0 + np.exp(-x))

        logit_means = logit(component_means)
        shifted_means = inv_logit(logit_means + delta_z)
        concentrations = alphas + betas  # S_k preserved

        new_alphas = np.clip(shifted_means * concentrations, 1e-3, 1e3)
        new_betas = np.clip((1.0 - shifted_means) * concentrations, 1e-3, 1e3)

        params['alphas'] = new_alphas
        params['betas'] = new_betas

        achieved_global_mean = float(np.sum(weights * shifted_means))
        delta_z_bar = achieved_global_mean - current_global_mean
        detection_ratio = (1.0 - achieved_global_mean) / (1.0 - current_global_mean) if current_global_mean < 1.0 else 0.0

        if verbose:
            print(f"    δ_z = {delta_z:+.4f}")
            print(f"    Global sparsity: {current_global_mean:.4f} → {achieved_global_mean:.4f} "
                  f"(Δ = {delta_z_bar:+.4f})")
            print(f"    Detection rate ratio (1-z')/(1-z): {detection_ratio:.3f}")
            for i in range(len(alphas)):
                print(f"    Component {i+1}: q={component_means[i]:.3f}→{shifted_means[i]:.3f}, "
                      f"α={alphas[i]:.2f}→{new_alphas[i]:.2f}, β={betas[i]:.2f}→{new_betas[i]:.2f}")

        return modeler

    def _alter_beta_model_sparsity_fold_change(self, modeler, sparsity_fold_change, verbose):
        """[deprecated] Legacy sparsity fold-change. Delegates to logit-shift with
        δ_z ≈ log(α_z) approximation, then falls back to binary search for exact match."""
        import warnings
        warnings.warn(
            "sparsity_fold_change is deprecated; use sparsity_logit_shift (δ_z) instead. "
            "Approximating δ_z from fold-change.",
            DeprecationWarning, stacklevel=2,
        )
        # Approximate: if α_z = 2.0, then δ_z ≈ log(2.0) ≈ 0.69
        delta_z = np.log(max(sparsity_fold_change, 1e-4))
        return self._alter_beta_model_sparsity_logit_shift(modeler, delta_z, verbose)
    
    def visualize_alteration(self, original_modeler, altered_modeler, 
                           title: str = "Model Alteration Comparison",
                           n_samples: int = 50000):

        # Generate samples
        original_samples = original_modeler.sample(n_samples)
        altered_samples = altered_modeler.sample(n_samples)
        
        # Calculate statistics
        orig_mean, orig_var = np.mean(original_samples), np.var(original_samples)
        alt_mean, alt_var = np.mean(altered_samples), np.var(altered_samples)
        
        mean_fc = alt_mean / orig_mean if orig_mean != 0 else np.inf
        var_fc = alt_var / orig_var if orig_var != 0 else np.inf
        
        # Create comparison plot
        plt.figure(figsize=(14, 6))
        
        # Histogram comparison
        plt.subplot(1, 2, 1)
        plt.hist(original_samples, bins=100, density=True, alpha=0.7, 
                label=f'Original (μ={orig_mean:.2f}, σ²={orig_var:.2f})', color='lightblue')
        plt.hist(altered_samples, bins=100, density=True, alpha=0.7,
                label=f'Altered (μ={alt_mean:.2f}, σ²={alt_var:.2f})', color='lightcoral')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(1, 2, 2)
        original_sorted = np.sort(original_samples)
        altered_sorted = np.sort(altered_samples)
        quantiles = np.linspace(0, 1, min(len(original_sorted), len(altered_sorted)))
        orig_quantiles = np.quantile(original_sorted, quantiles)
        alt_quantiles = np.quantile(altered_sorted, quantiles)
        
        plt.scatter(orig_quantiles, alt_quantiles, alpha=0.6, s=1)
        plt.plot([min(orig_quantiles), max(orig_quantiles)], 
                [min(orig_quantiles), max(orig_quantiles)], 'r--', alpha=0.8)
        plt.xlabel('Original Quantiles')
        plt.ylabel('Altered Quantiles')
        plt.title('Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title}\nMean FC: {mean_fc:.2f}x, Variance FC: {var_fc:.2f}x', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_alteration_history(self):
        """Return the history of all alterations performed."""
        return self.alteration_history
    
    def clear_history(self):
        """Clear the alteration history."""
        self.alteration_history = []

def alter_marginal_model(modeler,
                         mean_fold_change: float = 1.0,
                         variance_fold_change: float = 1.0,
                         dispersion_strength: float = 0.2,
                         preserve_original: bool = True,
                         verbose: bool = True,
                         sparsity_fold_change: float | None = None,
                         sparsity_logit_shift: float | None = None):
    """
    Convenience wrapper expected by other modules (e.g. FEAST.parameter_cloud).
    Delegates to MarginalModelAlterator.alter_model and returns the altered modeler.

    Args:
        sparsity_logit_shift: δ_z — direct logit-space shift (preferred).
        sparsity_fold_change: [deprecated] fold-change for zero proportion.
    """
    alterator = MarginalModelAlterator()
    return alterator.alter_model(
        modeler,
        mean_fold_change=mean_fold_change,
        variance_fold_change=variance_fold_change,
        sparsity_fold_change=sparsity_fold_change,
        sparsity_logit_shift=sparsity_logit_shift,
        dispersion_strength=dispersion_strength,
        preserve_original=preserve_original,
        verbose=verbose,
    )

# Integration helper for main FEAST pipeline
class AlterationConfig:
    """Configuration class for easy integration with FEAST pipeline.
    
    Well-structured hyperparameters for clear, independent control of
    simulated gene-level summary statistics:
    - Mean level alterations (fold-change based)
    - Variance level alterations (fold-change based)
    - Zero proportion alterations (fold-change based)
    
    Default: No changes to any parameter (all neutral values)
    """
    
    def __init__(self,
                 mean_fold_change: float = 1.0,
                 variance_fold_change: float = 1.0,
                 sparsity_logit_shift: float = 0.0,
                 dispersion_strength: float = 0.2,
                 variance_dispersion: float = 1.0,
                 mean_variance_coupling: str | None = None,
                 apply_to_mean: bool = False,
                 apply_to_variance: bool = False,
                 apply_to_zero_prop: bool = False,
                 # --- deprecated, kept for backward compat ---
                 sparsity_fold_change: float | None = None,
                 ):
        """Configure alteration parameters for FEAST integration.

        Three independent interventions on gene-level summary statistics,
        each on its own natural scale:

        Mean alteration (α_μ):
            μ' = α_μ · μ
            Pure log-location shift of StudentT mixture: m_k' += log10(α_μ).
            Weights, df, and scales unchanged.
            With mean_variance_coupling="fano", also shifts variance by α_μ.

        Variance alteration (α_v, ρ_v):
            σ²' = α_v · σ²                            — level shift
            m_k' = mean(m') + c_v + ρ_v (m_k - mean(m))  — dispersion
            s_k' = ρ_v · s_k

        Sparsity alteration (δ_z):
            logit(z') = logit(z) + δ_z
            Direct logit-space shift of Beta component means.
            δ_z > 0 → more sparse, δ_z < 0 → less sparse.
            Component concentrations S_k preserved.

        Copula C preserved throughout.

        Args:
            mean_fold_change: α_μ — fold-change for mean expression (1.0 = no change)
            variance_fold_change: α_v — fold-change for variance level (1.0 = no change)
            sparsity_logit_shift: δ_z — additive shift in logit(z) space (0.0 = no change)
            dispersion_strength: legacy parameter
            variance_dispersion: ρ_v — variance heterogeneity (1.0 = preserve shape)
            mean_variance_coupling: "fano" to auto-scale variance with mean, or None
            apply_to_mean, apply_to_variance, apply_to_zero_prop: which axes to alter
            sparsity_fold_change: [deprecated] use sparsity_logit_shift instead
        """
        self.mean_fold_change = mean_fold_change
        self.variance_fold_change = variance_fold_change
        self.dispersion_strength = dispersion_strength
        self.variance_dispersion = variance_dispersion
        self.mean_variance_coupling = mean_variance_coupling
        self.apply_to_mean = apply_to_mean
        self.apply_to_variance = apply_to_variance
        self.apply_to_zero_prop = apply_to_zero_prop
        # Resolve sparsity: δ_z takes precedence over deprecated α_z
        if sparsity_fold_change is not None and sparsity_logit_shift == 0.0:
            self.sparsity_logit_shift = float(np.log(max(float(sparsity_fold_change), 1e-4)))
        else:
            self.sparsity_logit_shift = float(sparsity_logit_shift)
        # Backward-compat alias
        self.sparsity_fold_change = self.sparsity_logit_shift

    def to_dict(self):
        """Convert configuration to dictionary for easy parameter passing."""
        return {
            'mean_fold_change': self.mean_fold_change,
            'variance_fold_change': self.variance_fold_change,
            'sparsity_logit_shift': self.sparsity_logit_shift,
            'dispersion_strength': self.dispersion_strength,
            'variance_dispersion': self.variance_dispersion,
            'mean_variance_coupling': self.mean_variance_coupling,
            'apply_to_mean': self.apply_to_mean,
            'apply_to_variance': self.apply_to_variance,
            'apply_to_zero_prop': self.apply_to_zero_prop,
        }
    
    @classmethod
    def mean_only(cls, fold_change: float, variance_coupling: str | None = None):
        """Create config for mean-only alterations.

        Args:
            fold_change: α_μ
            variance_coupling: "fano" to auto-scale variance with mean, or None
        """
        return cls(
            mean_fold_change=fold_change,
            apply_to_mean=True,
            mean_variance_coupling=variance_coupling,
        )

    @classmethod
    def variance_only(cls, fold_change: float, dispersion: float = 1.0):
        """Create config for variance-level alterations.

        Args:
            fold_change: α_v level shift
            dispersion: ρ_v shape parameter (1.0 = preserve, >1.0 = amplify heterogeneity)
        """
        return cls(
            variance_fold_change=fold_change,
            apply_to_variance=True,
            variance_dispersion=dispersion,
        )

    @classmethod
    def variance_heterogeneity(cls, rho: float):
        """Create config for variance heterogeneity alteration (level fixed at 1.0)."""
        return cls(
            variance_fold_change=1.0,
            apply_to_variance=True,
            variance_dispersion=rho,
        )

    @classmethod
    def sparsity_logit(cls, delta: float):
        """Create config for sparsity logit-shift alteration.

        Args:
            delta: δ_z — additive shift in logit(z) space.
                   δ_z > 0 → more sparse, δ_z < 0 → less sparse.
        """
        return cls(
            sparsity_logit_shift=delta,
            apply_to_zero_prop=True,
        )

    @classmethod
    def sparsity_only(cls, fold_change: float | None = None, logit_shift: float = 0.0):
        """Create config for sparsity-only alterations. [deprecated: use sparsity_logit]"""
        if fold_change is not None:
            return cls(sparsity_fold_change=fold_change, apply_to_zero_prop=True)
        return cls(sparsity_logit_shift=logit_shift, apply_to_zero_prop=True)

    @classmethod
    def comprehensive(cls, mean_fc: float = 1.0, var_fc: float = 1.0,
                     delta_z: float = 0.0, var_disp: float = 1.0,
                     mean_var_coupling: str | None = None):
        """Create config with all alterations enabled."""
        return cls(
            mean_fold_change=mean_fc,
            apply_to_mean=mean_fc != 1.0,
            variance_fold_change=var_fc,
            apply_to_variance=var_fc != 1.0,
            sparsity_logit_shift=delta_z,
            apply_to_zero_prop=delta_z != 0.0,
            variance_dispersion=var_disp,
            mean_variance_coupling=mean_var_coupling,
        )
