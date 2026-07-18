import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import betaln, digamma
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.cluster import KMeans

def calculate_aic_beta(log_likelihood, n_params):
    """Calculate AIC score for model selection."""
    return 2 * n_params - 2 * log_likelihood

def calculate_bic_beta(log_likelihood, n_params, n_samples):
    """Calculate BIC score for model selection."""
    return np.log(n_samples) * n_params - 2 * log_likelihood

def _beta_logpdf(x, alpha, beta_param):
    """Log-PDF of Beta distribution."""
    return beta.logpdf(x, alpha, beta_param)

def _beta_pdf(x, alpha, beta_param):
    """PDF of Beta distribution."""
    return beta.pdf(x, alpha, beta_param)

def _beta_rvs(alpha, beta_param, size=1, random_state=None):
    """Sample from Beta distribution."""
    return beta.rvs(alpha, beta_param, size=size, random_state=random_state)

def fit_single_beta(data):
    """
    Fit a single Beta distribution to data using method of moments.
    
    Args:
        data: Input data array (should be in [0, 1])
        
    Returns:
        dict: Parameters {'alpha': alpha, 'beta': beta}
    """
    # Remove invalid values and ensure data is in [0, 1]
    data_clean = data[(data > 0) & (data < 1) & np.isfinite(data)]
    if len(data_clean) == 0:
        return {'alpha': 1.0, 'beta': 1.0}
    
    # Method of moments for Beta distribution
    mean_val = np.mean(data_clean)
    var_val = np.var(data_clean)
    
    # Beta distribution: E[X] = a/(a+b), Var[X] = ab/((a+b)^2(a+b+1))
    if var_val > 0 and 0 < mean_val < 1:
        # Solve for alpha and beta using method of moments
        # mean = a/(a+b), var = ab/((a+b)^2(a+b+1))
        # Let s = a+b, then: mean = a/s, var = a(s-a)/(s^2(s+1))
        # var = mean(1-mean)/(s+1), so s = mean(1-mean)/var - 1
        sum_params = mean_val * (1 - mean_val) / var_val - 1
        alpha_est = mean_val * sum_params
        beta_est = (1 - mean_val) * sum_params
        
        # Ensure reasonable bounds
        alpha_est = max(0.1, min(alpha_est, 100))
        beta_est = max(0.1, min(beta_est, 100))
    else:
        # Fallback to uniform-like distribution
        alpha_est, beta_est = 1.0, 1.0
        
    return {'alpha': alpha_est, 'beta': beta_est}

def fit_beta_mixture_em(data, n_components=2, max_iter=200, tol=1e-6):
    """
    Fit Beta mixture using EM algorithm.
    
    Args:
        data: Input data (should be in [0, 1])
        n_components: Number of mixture components
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        
    Returns:
        dict: Model parameters and log-likelihood
    """
    # Clean data - ensure it's in (0, 1) for Beta distribution
    data = data[(data > 0) & (data < 1) & np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("No valid data points in (0, 1) for Beta mixture fitting.")
    
    n_samples = len(data)
    
    # Initialize using K-means on logit-transformed data
    logit_data = np.log(data / (1 - data)).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_components, n_init='auto', random_state=0).fit(logit_data)

    weights = np.array([np.sum(kmeans.labels_ == i) / n_samples for i in range(n_components)])
    alphas = np.ones(n_components)
    betas = np.ones(n_components)
    
    # Initialize parameters for each component
    for i in range(n_components):
        cluster_data = data[kmeans.labels_ == i]
        if len(cluster_data) > 1:
            params = fit_single_beta(cluster_data)
            alphas[i], betas[i] = params['alpha'], params['beta']
        else:
            alphas[i], betas[i] = 1.0, 1.0

    log_likelihood = -np.inf
    
    # EM algorithm
    for iteration in range(max_iter):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((n_samples, n_components))
        component_logpdfs = np.zeros((n_samples, n_components))

        log_weights = np.full(n_components, -np.inf, dtype=np.float64)
        positive_weight = weights > 0
        log_weights[positive_weight] = np.log(weights[positive_weight])
        for k in range(n_components):
            component_logpdfs[:, k] = log_weights[k] + _beta_logpdf(data, alphas[k], betas[k])
        
        # Numerical stability
        max_log = np.max(component_logpdfs, axis=1, keepdims=True)
        exp_logpdfs = np.exp(component_logpdfs - max_log)
        sum_exp = np.sum(exp_logpdfs, axis=1, keepdims=True)
        responsibilities = exp_logpdfs / sum_exp
        
        # Calculate log-likelihood
        log_sum_exp = max_log + np.log(sum_exp)
        new_log_likelihood = np.sum(log_sum_exp)
        
        # M-step: Update parameters
        weights = np.mean(responsibilities, axis=0)

        # Pre-compute once per EM iteration
        log_x = np.log(np.clip(data, 1e-15, 1 - 1e-15))
        log_1mx = np.log(np.clip(1 - data, 1e-15, 1 - 1e-15))

        for k in range(n_components):
            resp_k = responsibilities[:, k]
            w_sum = float(np.sum(resp_k))
            if w_sum < 1e-10:
                continue

            # Analytical gradient for Beta negative log-likelihood
            s1 = float(np.sum(resp_k * log_x) / w_sum)
            s2 = float(np.sum(resp_k * log_1mx) / w_sum)

            def objective(params):
                a, b = float(params[0]), float(params[1])
                if a <= 0.005 or b <= 0.005 or a > 200 or b > 200:
                    return np.inf
                # -log L = W * [betaln(a,b) - (a-1)*s1 - (b-1)*s2]
                return w_sum * (betaln(a, b) - (a - 1) * s1 - (b - 1) * s2)

            def gradient(params):
                a, b = float(params[0]), float(params[1])
                psi_ab = digamma(a + b)
                # d/da: W * [psi(a) - psi(a+b) - s1]
                g_a = w_sum * (digamma(a) - psi_ab - s1)
                # d/db: W * [psi(b) - psi(a+b) - s2]
                g_b = w_sum * (digamma(b) - psi_ab - s2)
                return np.array([g_a, g_b], dtype=np.float64)

            result = minimize(
                objective,
                x0=[float(alphas[k]), float(betas[k])],
                jac=gradient,
                bounds=[(0.01, 200), (0.01, 200)],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12},
            )
            if result.success:
                alphas[k], betas[k] = result.x
        
        # Check convergence
        if abs(new_log_likelihood - log_likelihood) < tol:
            log_likelihood = new_log_likelihood
            break
        log_likelihood = new_log_likelihood
    
    n_params = (n_components - 1) + 2 * n_components
    return {'weights': weights, 'alphas': alphas, 'betas': betas, 'n_params': n_params}, log_likelihood

class BetaMixtureMarginalModeler:
    """
    Beta Mixture Model for zero proportion modeling.
    
    This class fits a mixture of Beta distributions to data bounded between 0 and 1,
    making it ideal for modeling zero proportions in gene expression data.
    """
    
    def __init__(self, max_components=8):
        """
        Initialize the Beta mixture modeler.
        
        Args:
            max_components: Maximum number of mixture components to consider (default: 8)
        """
        self._is_fitted = False
        self.model_params = None
        self.log_transform = False  # Beta distribution doesn't need log transform
        self.max_components = max_components
        self._ppf_cache = {}
        self.best_n_components = None
        self.model_selection_scores = None
    
    def fit(self, data, visualize=True, selection_criterion='bic',
            early_stopping_patience: int = 2, n_jobs: int = 1):
        """Fit Beta mixture model to data with automatic component selection.

        Args:
            data: Input data (pandas Series or numpy array) — proportions in [0, 1]
            visualize: Whether to show fit visualization (default: True)
            selection_criterion: 'bic' or 'aic' for model selection (default: 'bic')
            early_stopping_patience: stop trying more components if the criterion
                has not improved for this many consecutive steps (0 = no early stop).
                Ignored when n_jobs > 1 (all components computed in parallel).
            n_jobs: Number of parallel workers for component search (1 = sequential).
        """
        self.log_transform = False

        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = np.array(data)

        processed_data = np.clip(data_array, 1e-10, 1 - 1e-10)
        processed_data = processed_data[np.isfinite(processed_data)]

        if len(processed_data) == 0:
            raise ValueError("No valid data points after preprocessing.")

        max_n = self.max_components
        n_samples = len(processed_data)

        if n_jobs > 1:
            # --- Parallel path: compute n=1..max_n in parallel ---
            from joblib import Parallel, delayed

            def _fit_one(n):
                if n == 1:
                    params = fit_single_beta(processed_data)
                    mp = {
                        'weights': np.array([1.0]),
                        'alphas': np.array([params['alpha']]),
                        'betas': np.array([params['beta']]),
                        'n_params': 2,
                    }
                    ll = np.sum(_beta_logpdf(processed_data, params['alpha'], params['beta']))
                else:
                    mp, ll = fit_beta_mixture_em(processed_data, n_components=n)
                if selection_criterion == 'bic':
                    sc = calculate_bic_beta(ll, mp['n_params'], n_samples)
                else:
                    sc = calculate_aic_beta(ll, mp['n_params'])
                return n, mp, ll, sc

            print(f"Fitting Beta mixture models with 1-{max_n} components"
                  f" (n_jobs={n_jobs})...")
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_fit_one)(n) for n in range(1, max_n + 1)
            )
            results.sort(key=lambda x: x[0])  # ensure component order

            models = [r[1] for r in results]
            log_likelihoods = [r[2] for r in results]
            scores = [r[3] for r in results]

            best_idx = np.argmin(scores)
            for n, _, ll, sc in results:
                flag = " *" if sc == scores[best_idx] else ""
                print(f"  {n} components: {selection_criterion.upper()}={sc:.2f}, "
                      f"LogLik={ll:.2f}{flag}")
        else:
            # --- Sequential path (original) ---
            n_components_range = range(1, max_n + 1)
            scores = []
            models = []
            log_likelihoods = []
            best_score = np.inf
            steps_since_best = 0

            print(f"Fitting Beta mixture models with 1-{max_n} components"
                  f" (early_stop patience={early_stopping_patience})...")

            for n in n_components_range:
                if n == 1:
                    params = fit_single_beta(processed_data)
                    model_params = {
                        'weights': np.array([1.0]),
                        'alphas': np.array([params['alpha']]),
                        'betas': np.array([params['beta']]),
                        'n_params': 2,
                    }
                    log_lik = np.sum(_beta_logpdf(processed_data, params['alpha'], params['beta']))
                else:
                    model_params, log_lik = fit_beta_mixture_em(processed_data, n_components=n)

                models.append(model_params)
                log_likelihoods.append(log_lik)

                if selection_criterion == 'bic':
                    score = calculate_bic_beta(log_lik, model_params['n_params'], len(processed_data))
                else:
                    score = calculate_aic_beta(log_lik, model_params['n_params'])

                scores.append(score)
                improved = score < best_score - 0.01
                flag = " *" if improved else ""
                print(f"  {n} components: {selection_criterion.upper()}={score:.2f}, LogLik={log_lik:.2f}{flag}")
                if improved:
                    best_score = score
                    steps_since_best = 0
                else:
                    steps_since_best += 1

                if early_stopping_patience > 0 and steps_since_best >= early_stopping_patience:
                    print(f"  Early stopping at {n} components "
                          f"(no improvement for {steps_since_best} steps).")
                    break

            best_idx = np.argmin(scores)
        self.best_n_components = best_idx + 1  # 0-indexed → component count
        self.model_params = models[best_idx]
        self.model_selection_scores = {
            'n_components': list(range(1, len(scores) + 1)),
            'scores': scores,
            'log_likelihoods': log_likelihoods,
            'best_idx': best_idx,
            'criterion': selection_criterion,
        }

        if self.model_params is None:
            raise RuntimeError("All model fitting attempts failed.")

        self._is_fitted = True

        print(f"✓ Best model: {self.best_n_components} components "
              f"({selection_criterion.upper()}={scores[best_idx]:.2f})")

        if visualize:
            self._visualize_fit(data, getattr(data, 'name', 'Data'))

        return self
    
    def sample(self, n, random_state=None):
        """
        Sample from the fitted Beta mixture model.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            numpy array of samples
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Choose components according to mixture weights
        rng = np.random if random_state is None else np.random.default_rng(int(random_state))
        component_choices = rng.choice(
            len(self.model_params['weights']),
            size=n,
            p=self.model_params['weights']
        )
        
        # Sample from chosen components
        samples = np.zeros(n)
        for i, (alpha, beta_param) in enumerate(zip(self.model_params['alphas'], self.model_params['betas'])):
            mask = component_choices == i
            n_samples = np.sum(mask)
            if n_samples > 0:
                samples[mask] = _beta_rvs(
                    alpha,
                    beta_param,
                    size=n_samples,
                    random_state=None if random_state is None else rng,
                )
        
        return samples
    
    def ppf(self, q, n_samples=200000, random_state=None):
        """
        Percent point function (inverse CDF) using sample-based approximation.
        
        Args:
            q: Quantile values (0-1)
            n_samples: Number of samples for approximation
            
        Returns:
            Quantile values
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Use cached samples if available
        cache_key = (
            n_samples
            if random_state is None
            else (n_samples, int(random_state))
        )
        if cache_key not in self._ppf_cache:
            samples = self.sample(n_samples, random_state=random_state)
            sorted_samples = np.sort(samples)
            quantiles = np.linspace(1/(n_samples+1), 1-1/(n_samples+1), n_samples)
            self._ppf_cache[cache_key] = (quantiles, sorted_samples)
        
        q_ref, v_ref = self._ppf_cache[cache_key]
        
        # Interpolate to get quantiles
        return np.interp(q, q_ref, v_ref)
    
    def pdf(self, x):
        """
        Probability density function of the fitted mixture model.
        
        Args:
            x: Input values (should be in [0, 1])
            
        Returns:
            PDF values
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        x = np.array(x)
        
        # Calculate mixture PDF
        pdf_vals = np.zeros_like(x)
        for weight, alpha, beta_param in zip(self.model_params['weights'], 
                                           self.model_params['alphas'], 
                                           self.model_params['betas']):
            pdf_vals += weight * _beta_pdf(x, alpha, beta_param)
        
        return pdf_vals
    
    def _visualize_fit(self, original_data, name):
        """
        Visualize the model fit.
        
        Args:
            original_data: Original data
            name: Name for plot title
        """
        if hasattr(original_data, 'values'):
            data_array = original_data.values
        else:
            data_array = np.array(original_data)
        
        # Clip data to [0, 1] for visualization
        plot_data = np.clip(data_array, 1e-10, 1 - 1e-10)
        plot_data = plot_data[np.isfinite(plot_data)]
        
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.hist(plot_data, bins=50, density=True, alpha=0.6, label="Original Data", color='lightgreen')
        
        # Plot fitted PDF
        x_range = np.linspace(0.001, 0.999, 1000)
        
        # Calculate mixture PDF
        pdf_vals = np.zeros_like(x_range)
        for weight, alpha, beta_param in zip(self.model_params['weights'], 
                                           self.model_params['alphas'], 
                                           self.model_params['betas']):
            pdf_vals += weight * _beta_pdf(x_range, alpha, beta_param)
        
        plt.plot(x_range, pdf_vals, 'r-', lw=2, 
                label=f"Fitted Beta Mixture ({self.best_n_components} components)")
        
        # Plot individual components if multiple
        if self.best_n_components > 1:
            for i, (weight, alpha, beta_param) in enumerate(zip(self.model_params['weights'], 
                                                              self.model_params['alphas'], 
                                                              self.model_params['betas'])):
                component_pdf = weight * _beta_pdf(x_range, alpha, beta_param)
                plt.plot(x_range, component_pdf, '--', alpha=0.7, 
                        label=f"Component {i+1} (w={weight:.2f}, α={alpha:.2f}, β={beta_param:.2f})")
        
        plt.title(f"Beta Mixture Fit for {name.title()}")
        plt.xlabel("Zero Proportion")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.show()
        
        # Print model summary
        print(f"\nModel Summary for {name}:")
        print(f"  Number of components: {self.best_n_components}")
        print(f"  Beta distribution (no log-transform needed)")
        for i in range(self.best_n_components):
            alpha = self.model_params['alphas'][i]
            beta_param = self.model_params['betas'][i]
            weight = self.model_params['weights'][i]
            mean = alpha / (alpha + beta_param)
            print(f"  Component {i+1}: weight={weight:.3f}, "
                  f"α={alpha:.3f}, β={beta_param:.3f}, mean={mean:.3f}")
