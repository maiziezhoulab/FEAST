import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.optimize import fsolve

warnings.filterwarnings('ignore')

def _qqplot_2samples_manual(data1, data2, xlabel, ylabel, line, ax):
    # 1. Ensure data are numpy arrays
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # 2. Determine the quantiles to plot (from 0.1% to 99.9%)
    n_points = min(len(data1), len(data2))
    percentiles = np.linspace(0.1, 99.9, num=max(100, n_points))

    # 3. Calculate the percentiles (quantiles) for each dataset
    quantiles1 = np.percentile(data1, percentiles)
    quantiles2 = np.percentile(data2, percentiles)

    # 4. Plot the quantiles against each other
    ax.scatter(quantiles1, quantiles2, alpha=0.5, s=15, c='dodgerblue', edgecolor='k', lw=0.5)

    # 5. Plot the reference line (y=x) for perfect agreement
    if line == '45':
        min_val = min(quantiles1.min(), quantiles2.min())
        max_val = max(quantiles1.max(), quantiles2.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    # 6. Set labels and improve aesthetics
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)


# =============================================================================
# STUDENT'S T MIXTURE MODEL CLASS (Now fully self-contained)
# =============================================================================
class StudentTMixtureMarginalModeler:
    """
    Student's T Mixture Model for modeling gene expression parameters.
    
    This approach uses a Gaussian Mixture Model for initialization and then
    refines the degrees of freedom for each component by fitting a t-distribution.
    """
    
    def __init__(self, max_components=8, random_state=42):
        self.max_components = max_components
        self.random_state = random_state
        self._is_fitted = False
        self.model_params = None
        
    def fit(self, data, log_transform=True, visualize=False):
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data) & (data > 0)]
        
        if len(data) < 10:
            raise ValueError("Insufficient data points for fitting")
        
        self.log_transform = log_transform
        self.data_range = (data.min(), data.max())
        processed_data = (np.log10(data + 1e-10) if self.log_transform else data).reshape(-1, 1)
        
        print(f"Fitting Student's T mixture models (1-{self.max_components} components)...")
        bics, models = [], []
        n_components_range = range(1, self.max_components + 1)
        
        for n in n_components_range:
            try:
                model = GaussianMixture(n_components=n, random_state=self.random_state, max_iter=200, n_init=3)
                model.fit(processed_data)
                bics.append(model.bic(processed_data))
                models.append(model)
            except Exception:
                bics.append(np.inf)
                models.append(None)
        
        best_idx = np.argmin(bics)
        gmm = models[best_idx]
        if gmm is None:
            raise RuntimeError("Failed to fit any Gaussian mixture model")
        
        self.model_params = {
            'n_components': n_components_range[best_idx],
            'weights': gmm.weights_,
            'means': gmm.means_.flatten(),
            'scales': np.sqrt(gmm.covariances_).flatten(),
            'dfs': np.zeros(gmm.n_components)
        }
        
        labels = gmm.predict(processed_data)
        for i in range(gmm.n_components):
            component_data = processed_data[labels == i].flatten()
            if len(component_data) > 10:
                try:
                    fitted_params = t.fit(component_data)
                    self.model_params['dfs'][i] = max(1.1, fitted_params[0])
                except:
                    self.model_params['dfs'][i] = 10.0
            else:
                self.model_params['dfs'][i] = 100.0
        
        self._is_fitted = True
        final_bic = bics[best_idx]
        print(f"✓ Best model: {self.model_params['n_components']} components (BIC: {final_bic:.2f})")
        
        if visualize:
            self._visualize_fit(data, "Gene Parameter")
        
        return {
            'n_components': self.model_params['n_components'],
            'weights': self.model_params['weights'],
            'means': self.model_params['means'],
            'scales': self.model_params['scales'],
            'dfs': self.model_params['dfs'],
            'bic': final_bic,
            'aic': final_bic - 2 * np.log(len(data))
        }
    
    def pdf(self, x):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        x_processed = np.log10(np.asarray(x) + 1e-10) if self.log_transform else np.asarray(x)
        pdf = np.zeros_like(x_processed, dtype=float)
        
        for i in range(self.model_params['n_components']):
            pdf += self.model_params['weights'][i] * t.pdf(x_processed, 
                df=self.model_params['dfs'][i], 
                loc=self.model_params['means'][i], 
                scale=self.model_params['scales'][i])
        return pdf
    
    def cdf(self, x):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        x_processed = np.log10(np.asarray(x) + 1e-10) if self.log_transform else np.asarray(x)
        cdf = np.zeros_like(x_processed, dtype=float)
        
        for i in range(self.model_params['n_components']):
            cdf += self.model_params['weights'][i] * t.cdf(x_processed, 
                df=self.model_params['dfs'][i], 
                loc=self.model_params['means'][i], 
                scale=self.model_params['scales'][i])
        return cdf
    
    def ppf(self, q):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        q = np.asarray(q)
        result = np.zeros_like(q, dtype=float)
        
        for i, quantile in enumerate(q.flatten()):
            if quantile <= 0: result.flat[i] = self.data_range[0]
            elif quantile >= 1: result.flat[i] = self.data_range[1]
            else:
                def equation(x): return self.cdf(x) - quantile
                x0 = 10**self.model_params['means'][0] if self.log_transform else self.model_params['means'][0]
                try:
                    solution = fsolve(equation, x0)[0]
                    result.flat[i] = max(solution, self.data_range[0])
                except:
                    result.flat[i] = np.interp(quantile, [0, 1], self.data_range)
        
        return result.reshape(q.shape)
    
    def sample(self, n_samples):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        component_choices = np.random.choice(self.model_params['n_components'], size=n_samples, p=self.model_params['weights'])
        samples = np.array([t.rvs(df=self.model_params['dfs'][c], loc=self.model_params['means'][c], scale=self.model_params['scales'][c]) for c in component_choices])
        
        return 10**samples if self.log_transform else samples
    
    def _visualize_fit(self, data, name):
        if not self._is_fitted:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        processed_data = np.log10(data + 1e-10) if self.log_transform else data
        
        # Plot 1: Density Fit
        ax1.hist(processed_data, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='Data')
        x_range = np.linspace(processed_data.min(), processed_data.max(), 1000)
        total_pdf = np.zeros_like(x_range)
        colors = plt.cm.viridis(np.linspace(0, 1, self.model_params['n_components']))
        for i in range(self.model_params['n_components']):
            w, df, loc, scale = (self.model_params['weights'][i], self.model_params['dfs'][i], self.model_params['means'][i], self.model_params['scales'][i])
            component_pdf = w * t.pdf(x_range, df=df, loc=loc, scale=scale)
            total_pdf += component_pdf
            ax1.plot(x_range, component_pdf, '--', color=colors[i], label=f'Comp {i+1} (w={w:.2f}, df={df:.1f})')
        ax1.plot(x_range, total_pdf, 'r-', linewidth=2.5, label=f'Fitted Mixture ({self.model_params["n_components"]} comps)')
        ax1.set_xlabel(f'log10({name})' if self.log_transform else name)
        ax1.set_ylabel('Density')
        ax1.set_title(f'Density Fit for {name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Q-Q Plot using our new, reliable manual function
        n_samples_for_qq = len(processed_data)
        component_choices = np.random.choice(self.model_params['n_components'], size=n_samples_for_qq, p=self.model_params['weights'])
        theoretical_samples = np.zeros(n_samples_for_qq)
        for i in range(n_samples_for_qq):
            comp_idx = component_choices[i]
            df, loc, scale = (self.model_params['dfs'][comp_idx], self.model_params['means'][comp_idx], self.model_params['scales'][comp_idx])
            theoretical_samples[i] = t.rvs(df=df, loc=loc, scale=scale)
        
        _qqplot_2samples_manual(
            data1=theoretical_samples,
            data2=processed_data,
            xlabel='Theoretical Quantiles (Fitted Model)',
            ylabel='Sample Quantiles (Original Data)',
            line='45',
            ax=ax2
        )
        ax2.set_title(f'Q-Q Plot of Fit vs. Data for {name}')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.suptitle(f"Student's T Mixture Model Fit Diagnostics", fontsize=18, y=1.02)
        plt.tight_layout()
        plt.show()

        print(f"\nStudent's T Mixture Components:")
        for i in range(self.model_params['n_components']):
            w, df, loc, scale = (self.model_params['weights'][i], self.model_params['dfs'][i], self.model_params['means'][i], self.model_params['scales'][i])
            print(f"  Component {i+1}: weight={w:.4f}, df={df:.2f}, loc={loc:.4f}, scale={scale:.4f}")