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
                   sparsity_fold_change: float = 1.0,
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
            
        if sparsity_fold_change <= 0:
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
            if sparsity_fold_change != 1.0:
                print(f"  Sparsity fold change: {sparsity_fold_change}x")
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
                                     variance_fold_change, sparsity_fold_change, verbose)
            else:
                raise ValueError(f"Unknown model parameter structure: {target_modeler.model_params.keys()}")
        else:
            raise ValueError("Modeler does not have recognizable model_params structure")
        
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
        
        # --- 1. Apply MEAN change ---
        if mean_fold_change != 1.0:
            if hasattr(modeler, 'log_transform') and modeler.log_transform:
                # On log scale, multiplicative change becomes additive shift
                log_mean_shift = np.log10(mean_fold_change)
                params['means'] += log_mean_shift
                if verbose:
                    print(f"  > Applied log10 mean shift: +{log_mean_shift:.3f}")
            else:
                # Direct multiplicative change
                params['means'] *= mean_fold_change
                if verbose:
                    print(f"  > Applied direct mean multiplier: {mean_fold_change}x")
        
        # --- 2. Apply VARIANCE change ---
        if variance_fold_change != 1.0:
            # Heuristic 1: Scale component standard deviations
            # Since Var(aX) = a²Var(X), we use sqrt of desired variance change
            if 'scales' in params:
                scale_inflation_factor = np.sqrt(variance_fold_change)
                params['scales'] *= scale_inflation_factor
                if verbose:
                    print(f"  > Scaled component widths by: {scale_inflation_factor:.3f}x")
            
            # Heuristic 2: Increase separation of component means
            # This creates additional variance through component dispersion
            if len(params['means']) > 1 and dispersion_strength > 0:
                mean_dispersion_factor = 1.0 + (variance_fold_change - 1.0) * dispersion_strength
                if mean_dispersion_factor != 1.0:
                    # Calculate weighted mean of current means
                    current_overall_mean = np.sum(params['weights'] * params['means'])
                    # Spread means around the overall mean
                    params['means'] = (current_overall_mean + 
                                     mean_dispersion_factor * (params['means'] - current_overall_mean))
                    if verbose:
                        print(f"  > Increased mean separation by: {mean_dispersion_factor:.3f}x")
    
    def _alter_beta_model(self, modeler, mean_fold_change, variance_fold_change, sparsity_fold_change, verbose):
        params = modeler.model_params
        
        if verbose:
            print("  > Altering Beta mixture model...")
        
        # Check if we're doing sparsity fold change (takes precedence over mean/var)
        if sparsity_fold_change != 1.0:
            return self._alter_beta_model_sparsity_fold_change(modeler, sparsity_fold_change, verbose)
        
        # Otherwise, do mean/variance fold-change based alterations
        # For Beta distribution: mean = α/(α+β), var = αβ/((α+β)²(α+β+1))
        # We adjust α and β to achieve desired mean and variance changes
        
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
    
    def _alter_beta_model_sparsity_fold_change(self, modeler, sparsity_fold_change, verbose):
        """
        Alter Beta mixture model using sparsity fold change.
        
        This method multiplies the current average sparsity by the fold change factor.
        """
        if sparsity_fold_change <= 0:
            raise ValueError("sparsity_fold_change must be positive")
            
        params = modeler.model_params
        weights = params['weights']
        alphas = params['alphas']
        betas = params['betas']
        
        # 1. Calculate current overall mean of the mixture
        component_means = alphas / (alphas + betas)
        current_overall_mean = np.sum(weights * component_means)
        
        # 2. Calculate target sparsity using fold change
        target_mean_sparsity = current_overall_mean * sparsity_fold_change
        
        # 3. Clip to valid range
        target_mean_sparsity = np.clip(target_mean_sparsity, 0.01, 0.99)
        
        if verbose:
            print(f"    Current Mean Sparsity: {current_overall_mean:.3f}")
            print(f"    Sparsity Fold Change: {sparsity_fold_change}x")
            print(f"    Target Mean Sparsity: {target_mean_sparsity:.3f}")
        
        # 4. Determine the required shift
        shift = target_mean_sparsity - current_overall_mean
        
        # 5. Apply shift to each component while preserving shape
        for i in range(len(alphas)):
            # Shift the mean of this component
            old_mean = component_means[i]
            new_mean = old_mean + shift
            
            # Clip to ensure the new mean is valid
            new_mean_clipped = np.clip(new_mean, 0.01, 0.99)
            
            # Preserve the sum S = alpha + beta to maintain component shape
            S = alphas[i] + betas[i]
            
            # Solve for the new alpha and beta
            new_alpha = new_mean_clipped * S
            new_beta = S - new_alpha
            
            # Ensure positive parameters
            new_alpha = max(0.1, new_alpha)
            new_beta = max(0.1, new_beta)
            
            params['alphas'][i] = new_alpha
            params['betas'][i] = new_beta
            
            if verbose:
                print(f"    Component {i+1}: α={alphas[i]:.2f}→{new_alpha:.2f}, β={betas[i]:.2f}→{new_beta:.2f}")
        
        # Verify final result
        final_component_means = params['alphas'] / (params['alphas'] + params['betas'])
        final_mean = np.sum(weights * final_component_means)
        achieved_fold_change = final_mean / current_overall_mean
        
        if verbose:
            print(f"    Achieved Mean Sparsity: {final_mean:.3f}")
            print(f"    Achieved Fold Change: {achieved_fold_change:.3f}x")
        
        return modeler
    
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
                         sparsity_fold_change: float = 1.0):
    """
    Convenience wrapper expected by other modules (e.g. FEAST.parameter_cloud).
    Delegates to MarginalModelAlterator.alter_model and returns the altered modeler.
    """
    alterator = MarginalModelAlterator()
    return alterator.alter_model(
        modeler,
        mean_fold_change=mean_fold_change,
        variance_fold_change=variance_fold_change,
        sparsity_fold_change=sparsity_fold_change,
        dispersion_strength=dispersion_strength,
        preserve_original=preserve_original,
        verbose=verbose
    )

# Integration helper for main FEAST pipeline
class AlterationConfig:
    """Configuration class for easy integration with FEAST pipeline.
    
    Well-structured hyperparameters for clear, independent control of:
    - Mean alterations (fold-change based)
    - Variance alterations (fold-change based) 
    - Zero proportion alterations (fold-change based)
    
    Default: No changes to any parameter (all neutral values)
    """
    
    def __init__(self, 
                 mean_fold_change: float = 1.0,
                 variance_fold_change: float = 1.0,
                 sparsity_fold_change: float = 1.0,
                 dispersion_strength: float = 0.2,
                 apply_to_mean: bool = False,
                 apply_to_variance: bool = False,
                 apply_to_zero_prop: bool = False):
        """
        Configure alteration parameters for FEAST integration.
        
        Args:
            mean_fold_change: Multiply gene expression means by this factor (1.0 = no change)
            variance_fold_change: Multiply gene expression variance by this factor (1.0 = no change)
            sparsity_fold_change: Multiply zero proportion by this factor (1.0 = no change)
                                - 0.5 = reduce sparsity by half (more expression)
                                - 2.0 = double sparsity (more zeros)
            dispersion_strength: Control strength of alterations (0-1)
            apply_to_mean: Enable mean alterations (default: False)
            apply_to_variance: Enable variance alterations (default: False)
            apply_to_zero_prop: Enable zero proportion alterations (default: False)
        """
        self.mean_fold_change = mean_fold_change
        self.variance_fold_change = variance_fold_change
        self.sparsity_fold_change = sparsity_fold_change
        self.dispersion_strength = dispersion_strength
        self.apply_to_mean = apply_to_mean
        self.apply_to_variance = apply_to_variance
        self.apply_to_zero_prop = apply_to_zero_prop
    
    def to_dict(self):
        """Convert configuration to dictionary for easy parameter passing."""
        return {
            'mean_fold_change': self.mean_fold_change,
            'variance_fold_change': self.variance_fold_change,
            'sparsity_fold_change': self.sparsity_fold_change,
            'dispersion_strength': self.dispersion_strength,
            'apply_to_mean': self.apply_to_mean,
            'apply_to_variance': self.apply_to_variance,
            'apply_to_zero_prop': self.apply_to_zero_prop
        }
    
    @classmethod
    def mean_only(cls, fold_change: float, strength: float = 0.2):
        """Create config for mean-only alterations."""
        return cls(
            mean_fold_change=fold_change,
            apply_to_mean=True,
            dispersion_strength=strength
        )
    
    @classmethod 
    def variance_only(cls, fold_change: float, strength: float = 0.2):
        """Create config for variance-only alterations."""
        return cls(
            variance_fold_change=fold_change,
            apply_to_variance=True,
            dispersion_strength=strength
        )
    
    @classmethod
    def sparsity_only(cls, fold_change: float, strength: float = 0.2):
        """Create config for sparsity-only alterations."""
        return cls(
            sparsity_fold_change=fold_change,
            apply_to_zero_prop=True,
            dispersion_strength=strength
        )
    
    @classmethod
    def comprehensive(cls, mean_fc: float = 1.0, var_fc: float = 1.0, 
                     sparsity_fc: float = 1.0, strength: float = 0.2):
        """Create config with all alterations enabled."""
        return cls(
            mean_fold_change=mean_fc,
            apply_to_mean=mean_fc != 1.0,
            variance_fold_change=var_fc,
            apply_to_variance=var_fc != 1.0,
            sparsity_fold_change=sparsity_fc,
            apply_to_zero_prop=sparsity_fc != 1.0,
            dispersion_strength=strength
        )
