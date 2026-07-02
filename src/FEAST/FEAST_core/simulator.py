import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import cdist

from .count_decoding import decode_counts_by_spatial_intensity
from .parameter_cloud import (
    BatchDeformation,
    GeneParameterSimulator,
    alteration_config_to_dict,
    calculate_fold_change,
    convert_params_for_new_simulator,
    resolve_simulation_mode,
)

PARAMETER_MODES = ("hungarian", "reference_stats")
SPATIAL_MODES = ("reference_rank", "ot_spatial")
MAX_DENSE_OT_PAIRS = 25_000_000   # ~200 MB as float64; switches to block OT above this
OT_MEMORY_BYTES_PER_PAIR = 64     # cost, plan, Sinkhorn kernel/work arrays, and temporaries

# Internal translation: public parameter_mode ↔ internal simulation_mode
_PARAMETER_TO_SIMULATION = {"hungarian": "generative", "reference_stats": "empirical"}


def _translate_parameter_mode(parameter_mode):
    """Translate public parameter_mode to internal simulation_mode string."""
    if parameter_mode in _PARAMETER_TO_SIMULATION:
        return _PARAMETER_TO_SIMULATION[parameter_mode]
    raise ValueError(f"parameter_mode must be one of {PARAMETER_MODES}, got '{parameter_mode}'")


def _translate_spatial_mode(spatial_mode):
    """Validate and normalize spatial_mode."""
    if spatial_mode not in SPATIAL_MODES:
        raise ValueError(f"spatial_mode must be one of {SPATIAL_MODES}, got '{spatial_mode}'")
    return spatial_mode


def _resolve_ot_block_max_pairs(max_block_pairs=None, memory_budget_gb=None):
    """Resolve optional OT block controls to a maximum dense pair count."""
    limits = []
    if max_block_pairs is not None:
        max_block_pairs = int(max_block_pairs)
        if max_block_pairs < 1:
            raise ValueError("ot_block_max_pairs must be a positive integer.")
        limits.append(max_block_pairs)

    if memory_budget_gb is not None:
        memory_budget_gb = float(memory_budget_gb)
        if not np.isfinite(memory_budget_gb) or memory_budget_gb <= 0:
            raise ValueError("ot_memory_budget_gb must be a positive finite number.")
        budget_bytes = memory_budget_gb * (1024 ** 3)
        limits.append(max(1, int(budget_bytes // OT_MEMORY_BYTES_PER_PAIR)))

    if not limits:
        return None
    return min(limits)


def _split_indices_by_coordinate(indices, coords, axis, midpoint=None):
    """Split indices spatially, falling back to a stable count split."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= 1:
        return indices, np.empty(0, dtype=np.int64)

    if midpoint is not None:
        left_mask = coords[indices, axis] <= midpoint
        left = indices[left_mask]
        right = indices[~left_mask]
        if left.size and right.size:
            return left, right

    order = np.argsort(coords[indices, axis], kind="mergesort")
    split_at = indices.size // 2
    return indices[order[:split_at]], indices[order[split_at:]]


def _iter_bounded_ot_subblocks(src_idx, tgt_idx, source_coords, target_coords, max_block_pairs):
    """Yield spatial sub-blocks whose dense source-target pair count is bounded."""
    if max_block_pairs is None:
        yield np.asarray(src_idx, dtype=np.int64), np.asarray(tgt_idx, dtype=np.int64)
        return

    stack = [(np.asarray(src_idx, dtype=np.int64), np.asarray(tgt_idx, dtype=np.int64))]
    while stack:
        cur_src, cur_tgt = stack.pop()
        if cur_src.size == 0 or cur_tgt.size == 0:
            continue

        pair_count = int(cur_src.size) * int(cur_tgt.size)
        if pair_count <= max_block_pairs or (cur_src.size <= 1 and cur_tgt.size <= 1):
            yield cur_src, cur_tgt
            continue

        combined_coords = np.vstack((source_coords[cur_src, :2], target_coords[cur_tgt, :2]))
        axis = int(np.argmax(np.ptp(combined_coords, axis=0)))
        midpoint = float(np.median(combined_coords[:, axis]))

        if cur_src.size > 1 and cur_tgt.size > 1:
            src_left, src_right = _split_indices_by_coordinate(cur_src, source_coords, axis, midpoint)
            tgt_left, tgt_right = _split_indices_by_coordinate(cur_tgt, target_coords, axis, midpoint)
            stack.append((src_right, tgt_right))
            stack.append((src_left, tgt_left))
        elif cur_src.size > 1:
            src_left, src_right = _split_indices_by_coordinate(cur_src, source_coords, axis, midpoint)
            stack.append((src_right, cur_tgt))
            stack.append((src_left, cur_tgt))
        else:
            tgt_left, tgt_right = _split_indices_by_coordinate(cur_tgt, target_coords, axis, midpoint)
            stack.append((cur_src, tgt_right))
            stack.append((cur_src, tgt_left))




def safe_calculate_qc_metrics(adata, verbose=False):
    try:
        if adata.n_vars > 0 and adata.n_obs > 0:
            sc.pp.calculate_qc_metrics(adata, percent_top=[20, 50, 100] if adata.n_vars > 100 else [50], inplace=True, log1p=False)
    except (ValueError, TypeError, ImportError, IndexError) as e:
        if verbose:
            print(f"Warning: QC calculation failed ({e}), using basic metrics only")
        adata.obs['total_counts'] = np.asarray(adata.X.sum(axis=1)).flatten()
        adata.obs['n_genes_by_counts'] = np.asarray((adata.X > 0).sum(axis=1)).flatten()
        adata.var['total_counts'] = np.asarray(adata.X.sum(axis=0)).flatten()
        adata.var['n_cells_by_counts'] = np.asarray((adata.X > 0).sum(axis=0)).flatten()

def _dense_matrix(X, dtype=None):
    if hasattr(X, 'toarray'):
        X = X.toarray()
    arr = np.asarray(X)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _gene_stats_from_matrix(matrix, gene_names) -> pd.DataFrame:
    from scipy.sparse import issparse
    if issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
        X_sq = matrix.copy()
        X_sq.data **= 2
        variances = np.asarray(X_sq.mean(axis=0)).ravel() - means ** 2
        n_obs = matrix.shape[0]
        nz = matrix.getnnz(axis=0) if hasattr(matrix, 'getnnz') else np.diff(matrix.tocsc().indptr)
        zero_prop = 1.0 - nz / n_obs
        return pd.DataFrame({
            'mean': means,
            'variance': variances,
            'zero_prop': zero_prop,
        }, index=pd.Index(gene_names, name='gene_id')).clip(lower=1e-10)
    mat = np.asarray(matrix, dtype=np.float64)
    n_obs = mat.shape[0]
    return pd.DataFrame({
        'mean': np.mean(mat, axis=0),
        'variance': np.var(mat, axis=0),
        'zero_prop': 1 - (np.count_nonzero(mat, axis=0) / n_obs),
    }, index=pd.Index(gene_names, name='gene_id')).clip(lower=1e-10)


def _model_selection_counts(model_params: dict) -> dict:
    selected = np.asarray(model_params.get('model_selected', []), dtype=object)
    return {str(model): int(np.sum(selected == model)) for model in sorted(set(selected.tolist()))}


def _hdf5_safe_metadata(value):
    if value is None:
        return "none"
    if isinstance(value, dict):
        return {str(k): _hdf5_safe_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hdf5_safe_metadata(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_parameter_cloud_fitting(adata, visualize_fits=False, use_heuristic_search=True, min_accepted_error=0.5, assignment_weights=None, screening_pool_size=100, top_n_to_fully_evaluate=10, n_jobs=-1, alteration_config=None, simulation_mode='generative', spatial_mode='reference_rank', assignment_method='hybrid', random_seed=None, hybrid_alpha=0.2, use_distributional_alteration=False, ppf_method='interp', beta_n_jobs=1, beta_early_stopping_patience: int = 2, assignment_solver='scipy', assignment_n_jobs=1, assignment_blocks=True, assignment_block_size=None, assignment_block_multiplier=8, convert_n_jobs=1):
    """
    Build an integrated FEAST gene-parameter table and convert it to count-model parameters.

    Args:
        alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions
        hybrid_alpha (float): Weight for log-space distance in hybrid OT cost (0.2 = 20% log, 80% raw).
                              Set to 1.0 for old pure-log-space behavior.
        use_distributional_alteration (bool): If True, alter marginal model parameters (θ → θ')
            before sampling.  If False (default), apply scalar fold-change after sampling (legacy).
        ppf_method (str): 'interp' (fast, default) or 'exact' (bit-identical fsolve)
        beta_n_jobs (int): parallel workers for Beta component search (1=sequential)
        beta_early_stopping_patience (int): patience for Beta mixture early stopping (0 = no early stop)
        assignment_blocks (bool): use batched Hungarian assignment. Set False for
            the original full-matrix global Hungarian assignment.
        convert_n_jobs (int): parallel workers for count-model parameter conversion.
    """
    print("\n>>> Entering STANDARD fitting pipeline: parameter_cloud <<<")

    if assignment_weights is None:
        assignment_weights = {'mean': 3, 'variance': 1, 'zero_prop': 1.0}
    mode = resolve_simulation_mode(simulation_mode)
    if use_heuristic_search:
        warnings.warn(
            "use_heuristic_search is retained for compatibility but is ignored by the "
            "integrated simulation_mode pipeline; generative mode uses Copula-rank OT.",
            RuntimeWarning,
            stacklevel=2,
        )

    simulator = GeneParameterSimulator(
        ppf_method=ppf_method,
        beta_n_jobs=beta_n_jobs,
        beta_early_stopping_patience=beta_early_stopping_patience,
    )
    simulator.hybrid_alpha = hybrid_alpha
    if mode == 'empirical':
        simulator.fit_statistics_only(adata)
    else:
        simulator.fit(adata, visualize_fits=visualize_fits)

    assigned_synthetic_params, diagnostics = simulator.build_gene_parameter_table(
        alteration_config=alteration_config,
        simulation_mode=mode,
        assignment_weights=assignment_weights,
        random_seed=random_seed,
        assignment_method=assignment_method,
        verbose=True,
        use_distributional_alteration=use_distributional_alteration,
        assignment_solver=assignment_solver,
        assignment_n_jobs=assignment_n_jobs,
        assignment_blocks=assignment_blocks,
        assignment_block_size=assignment_block_size,
        assignment_block_multiplier=assignment_block_multiplier,
    )

    model_params = convert_params_for_new_simulator(
        assigned_synthetic_params, n_spots=adata.n_obs, n_jobs=convert_n_jobs)
    model_params['simulation_evaluation'] = {
        'source': 'integrated_parameter_cloud',
        'simulation_mode': mode,
    }
    model_params['simulation_mode'] = mode
    model_params['random_seed'] = None if random_seed is None else int(random_seed)
    model_params['target_stats'] = assigned_synthetic_params
    model_params['parameter_diagnostics'] = diagnostics
    print(">>> Exiting parameter_cloud pipeline <<<\n")
    return model_params

def run_direct_fitting_from_real_stats(adata):
    """Run diagnostic pipeline using real stats directly."""
    print("\n>>> Entering DIAGNOSTIC fitting pipeline: Using REAL stats directly <<<")
    simulator = GeneParameterSimulator()
    simulator.fit_statistics_only(adata)
    real_stats_for_conversion = simulator.original_stats.reset_index().rename(columns={'index': 'gene_id'})
    model_params = convert_params_for_new_simulator(
        real_stats_for_conversion, n_spots=adata.n_obs)
    model_params['simulation_evaluation'] = {'source': 'direct_from_real_stats', 'simulation_mode': 'empirical'}
    model_params['simulation_mode'] = 'empirical'
    model_params['target_stats'] = real_stats_for_conversion
    print(">>> Diagnostic fitting complete <<<\n")
    return model_params


def simulate_batch_effect(
    adata_ref,
    D: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    *,
    random_seed=None,
    boundary_multiplier: float = 1.1,
) -> "ad.AnnData":
    """Simulate a batch-affected slice from a reference AnnData.

    Pipeline:
      1. Extract per-gene stats from reference   -> _gene_stats_from_matrix()
      2. Convert stats to theta                   -> stats_to_theta()
      3. Apply affine deformation                 -> apply_batch_deformation()
      4. Convert back to stats                    -> theta_to_stats()
      5. Convert stats to count-model params      -> convert_params_for_new_simulator()
      6. Decode counts from reference intensity   -> decode_counts_by_spatial_intensity()

    Parameters
    ----------
    adata_ref : AnnData   -- reference (clean) slice
    D : (3,) ndarray      -- diagonal scaling coefficients
    b : (3,) ndarray      -- shift vector
    alpha : float         -- interpolation strength (0 = no effect, 1 = full)
    random_seed : int, optional
    boundary_multiplier : float

    Returns
    -------
    adata_sim : AnnData   -- batch-affected slice with preserved spatial pattern
    """
    from scipy.sparse import issparse
    from .theta_transform import stats_to_theta, theta_to_stats
    from .parameter_cloud import apply_batch_deformation, convert_params_for_new_simulator
    from .count_decoding import decode_counts_by_spatial_intensity

    ref_matrix = adata_ref.X.toarray() if issparse(adata_ref.X) else np.asarray(
        adata_ref.X, dtype=np.float64
    )
    n_obs = ref_matrix.shape[0]
    gene_names = list(adata_ref.var_names)

    stats_ref = _gene_stats_from_matrix(ref_matrix, gene_names)
    theta_ref = stats_to_theta(stats_ref)
    theta_batch = apply_batch_deformation(theta_ref, D, b, alpha)
    stats_batch = theta_to_stats(theta_batch).clip(lower=1e-10)

    stats_for_conversion = stats_batch.copy()
    stats_for_conversion.index = gene_names
    stats_for_conversion = stats_for_conversion.reset_index().rename(
        columns={"index": "gene_id"}
    )
    model_params = convert_params_for_new_simulator(
        stats_for_conversion, n_spots=n_obs, boundary_multiplier=boundary_multiplier
    )
    model_params["simulation_mode"] = "empirical"
    model_params["target_stats"] = stats_batch[["mean", "variance", "zero_prop"]].reset_index(drop=True)
    theta_delta = np.abs(theta_batch - theta_ref)
    model_params["parameter_diagnostics"] = {
        "requested_config": {
            "apply_to_variance": bool(np.any(theta_delta[:, 1] > 1e-8)),
            "apply_to_zero_prop": bool(np.any(theta_delta[:, 2] > 1e-8)),
            "mean_variance_coupling": None,
        }
    }

    simulated_matrix = decode_counts_by_spatial_intensity(
        ref_matrix.astype(np.float64),
        model_params,
        boundary_multiplier=boundary_multiplier,
        reference_X=ref_matrix,
        random_seed=random_seed,
    )

    sim_adata = ad.AnnData(
        X=simulated_matrix.astype(np.float32),
        obs=adata_ref.obs.copy(),
        var=adata_ref.var.copy(),
        obsm={k: v.copy() for k, v in adata_ref.obsm.items()},
    )
    if hasattr(adata_ref, "uns") and adata_ref.uns:
        sim_adata.uns = adata_ref.uns.copy()
    else:
        sim_adata.uns = {}

    sim_adata.uns["batch_deformation"] = {
        "D": D.tolist(),
        "b": b.tolist(),
        "alpha": alpha,
    }
    sim_adata.var["theta_mu_ref"] = theta_ref[:, 0]
    sim_adata.var["theta_omega_ref"] = theta_ref[:, 1]
    sim_adata.var["theta_pi0_ref"] = theta_ref[:, 2]
    sim_adata.var["theta_mu_batch"] = theta_batch[:, 0]
    sim_adata.var["theta_omega_batch"] = theta_batch[:, 1]
    sim_adata.var["theta_pi0_batch"] = theta_batch[:, 2]
    return sim_adata


def characterize_batch(
    adata_ref,
    adata_query,
    *,
    name: str = "",
) -> "BatchDeformation":
    """Estimate batch deformation (D, b) from two real slices via OLS in theta space.

    For each dimension k in {log_mu, log_omega, logit_pi0}:
        theta_query[:, k] = d_k * theta_ref[:, k] + b_k  (+ epsilon)

    This is the empirical-characterisation counterpart to simulate_batch_effect().
    Once (D, b) are estimated, they can be fed to simulate_batch_effect() with
    alpha in [0, 1] to generate interpolated batch-effect ladder.

    Parameters
    ----------
    adata_ref : AnnData    -- reference (clean) slice
    adata_query : AnnData  -- batch-affected slice
    name : str             -- optional label for the deformation

    Returns
    -------
    BatchDeformation with additional diagnostic attributes:
        .r2          : (3,) ndarray  — per-dimension R²
        .residual_std: (3,) ndarray  — per-dimension residual std dev
        .n_genes     : int           — number of common genes used
    """
    from scipy.sparse import issparse
    from .theta_transform import stats_to_theta

    X_ref = adata_ref.X.toarray() if issparse(adata_ref.X) else np.asarray(
        adata_ref.X, dtype=np.float64
    )
    X_qry = adata_query.X.toarray() if issparse(adata_query.X) else np.asarray(
        adata_query.X, dtype=np.float64
    )

    # Restrict to genes present in both
    common_mask = np.intersect1d(adata_ref.var_names, adata_query.var_names, return_indices=True)
    gene_idx_r, gene_idx_q = common_mask[1], common_mask[2]
    if len(gene_idx_r) < 3:
        raise ValueError(f"Fewer than 3 common genes found ({len(gene_idx_r)}).")

    stats_ref = _gene_stats_from_matrix(
        X_ref[:, gene_idx_r], list(adata_ref.var_names[gene_idx_r])
    )
    stats_qry = _gene_stats_from_matrix(
        X_qry[:, gene_idx_q], list(adata_query.var_names[gene_idx_q])
    )

    theta_ref = stats_to_theta(stats_ref)
    theta_qry = stats_to_theta(stats_qry)

    D = np.ones(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    r2 = np.zeros(3, dtype=np.float64)
    resid_std = np.zeros(3, dtype=np.float64)

    for k in range(3):
        x = theta_ref[:, k]
        y = theta_qry[:, k]

        ss_xx = np.sum((x - x.mean()) ** 2)
        if ss_xx < 1e-15:
            D[k], b[k] = 1.0, y.mean() - x.mean()
            r2[k] = 0.0
            resid_std[k] = float(np.std(y - y.mean()))
        else:
            D[k] = np.sum((x - x.mean()) * (y - y.mean())) / ss_xx
            b[k] = y.mean() - D[k] * x.mean()
            y_pred = D[k] * x + b[k]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2[k] = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-15)))
            resid_std[k] = float(np.sqrt(ss_res / len(y)))

    deform = BatchDeformation(D, b, alpha=1.0, name=name)
    deform.r2 = r2
    deform.residual_std = resid_std
    deform.n_genes = len(gene_idx_r)
    return deform


class SpatialSimulator:
    def __init__(self, reference_adata: ad.AnnData, model_params: dict = None):
        if 'spatial' not in reference_adata.obsm: raise ValueError("Reference AnnData must contain 'spatial' coordinates.")
        self.reference_adata = reference_adata.copy() 
        self.reference_adata.var_names_make_unique()
        self.reference_adata.obs_names_make_unique()
        self._model_params = model_params

    def fit_model(self, visualize_fits: bool = False, use_real_stats_directly: bool = False, use_heuristic_search: bool = False, min_accepted_error: float = 0.5, assignment_weights: dict = None, screening_pool_size: int = 100, top_n_to_fully_evaluate: int = 10, n_jobs: int = -1, alteration_config=None, simulation_mode: str = 'generative', spatial_mode: str = 'reference_rank', assignment_method: str = 'hybrid', random_seed: int = None, hybrid_alpha: float = 0.2, use_distributional_alteration: bool = False, ppf_method: str = 'interp', beta_n_jobs: int = 1, beta_early_stopping_patience: int = 2, assignment_solver: str = 'scipy', assignment_n_jobs: int = 1, assignment_blocks: bool = True, assignment_block_size: int = None, assignment_block_multiplier: int = 8, convert_n_jobs: int = 1) -> 'SpatialSimulator':
        """
        Exposes heuristic search parameters and marginal distribution alteration.

        Args:
            alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions
            use_distributional_alteration (bool): If True, alter marginal model parameters before sampling.
            ppf_method (str): 'interp' (fast) or 'exact' (bit-identical fsolve)
            beta_n_jobs (int): parallel workers for Beta component search
            beta_early_stopping_patience (int): patience for Beta mixture early stopping (0 = no early stop)
            assignment_blocks (bool): use batched Hungarian assignment. Set False
                for the original full-matrix global Hungarian assignment.
            convert_n_jobs (int): parallel workers for count-model conversion.
        """
        adata_for_fitting = self.reference_adata.copy(); safe_calculate_qc_metrics(adata_for_fitting)
        if use_real_stats_directly:
            self._model_params = run_direct_fitting_from_real_stats(adata_for_fitting)
        else:
            self._model_params = run_parameter_cloud_fitting(
                adata_for_fitting,
                visualize_fits=visualize_fits,
                use_heuristic_search=use_heuristic_search,
                min_accepted_error=min_accepted_error,
                assignment_weights=assignment_weights,
                screening_pool_size=screening_pool_size,
                top_n_to_fully_evaluate=top_n_to_fully_evaluate,
                n_jobs=n_jobs,
                alteration_config=alteration_config,
                simulation_mode=simulation_mode,
                spatial_mode=spatial_mode,
                assignment_method=assignment_method,
                random_seed=random_seed,
                hybrid_alpha=hybrid_alpha,
                use_distributional_alteration=use_distributional_alteration,
                ppf_method=ppf_method,
                beta_n_jobs=beta_n_jobs,
                beta_early_stopping_patience=beta_early_stopping_patience,
                assignment_solver=assignment_solver,
                assignment_n_jobs=assignment_n_jobs,
                assignment_blocks=assignment_blocks,
                assignment_block_size=assignment_block_size,
                assignment_block_multiplier=assignment_block_multiplier,
                convert_n_jobs=convert_n_jobs,
            )
        return self
    
    def set_model_params(self, model_params: dict):
        """Set model parameters directly."""
        self._model_params = model_params
        return self
    
    def get_model_params(self):
        """Get current model parameters."""
        return self._model_params
    
    def simulate(self, num_simulation_cores: int = 12, verbose: bool = True, clip_overshoot_factor: float = 0.0, boundary_multiplier: float = 1.1, random_seed: int = None, spatial_mode: str = 'reference_rank', target_adata=None, ot_block_size: int = 40000, ot_overlap_frac: float = 0.25, ot_block_max_pairs: int = None, ot_memory_budget_gb: float = None) -> ad.AnnData:
        """
        Args:
            num_simulation_cores (int): Number of cores for simulation (legacy parameter).
            verbose (bool): If True, prints progress updates.
            clip_overshoot_factor (float): Factor to clip max expression values relative to reference.
            boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max).
            random_seed (int, optional): Seed for reproducible sampling.
            spatial_mode: 'reference_rank' or 'ot_spatial'.
            target_adata: Required when spatial_mode='ot_spatial'.
            ot_block_size: Target spots per OT spatial grid tile.
            ot_overlap_frac: Fractional spatial overlap between OT grid tiles.
            ot_block_max_pairs: Optional hard cap for dense source-target pairs
                in each OT sub-block. Defaults to None, preserving existing
                block behavior.
            ot_memory_budget_gb: Optional memory budget converted to an OT
                sub-block pair cap. Defaults to None.
        """
        if self._model_params is None:
            raise ValueError("Model parameters not set. Call fit_model() first or provide model_params in constructor.")

        if verbose:
            print("Generating simulated data with quantile count decoding...")

        simulated_adata = self._apply_quantile_count_decoding(
            reference_adata=self.reference_adata,
            model_params=self._model_params,
            verbose=verbose,
            clip_overshoot_factor=clip_overshoot_factor,
            boundary_multiplier=boundary_multiplier,
            random_seed=random_seed,
            spatial_mode=spatial_mode,
            target_adata=target_adata,
            ot_block_size=ot_block_size,
            ot_overlap_frac=ot_overlap_frac,
            ot_block_max_pairs=ot_block_max_pairs,
            ot_memory_budget_gb=ot_memory_budget_gb,
        )
        
        if verbose:
            print(f"Quantile simulation complete for {simulated_adata.n_obs} spots and {simulated_adata.n_vars} genes")
        
        safe_calculate_qc_metrics(simulated_adata)
        return simulated_adata

    def _apply_quantile_count_decoding(self, reference_adata, model_params, verbose=True, clip_overshoot_factor=0.0, boundary_multiplier=1.1, random_seed=None, spatial_mode='reference_rank', target_adata=None, ot_block_size=40000, ot_overlap_frac=0.25, ot_block_max_pairs=None, ot_memory_budget_gb=None):
        """Generate counts from model parameters through rank-based count decoding."""
        from scipy.sparse import issparse
        reference_sparse = reference_adata.X if issparse(reference_adata.X) else None
        reference_matrix = _dense_matrix(reference_adata.X, dtype=np.float64)
        n_spots, n_genes = reference_matrix.shape

        mode = resolve_simulation_mode(model_params.get('simulation_mode', 'empirical'))
        diagnostics_seed = model_params.get('random_seed', None)
        seed = random_seed if random_seed is not None else diagnostics_seed
        if spatial_mode not in SPATIAL_MODES:
            raise ValueError(f"spatial_mode must be one of {SPATIAL_MODES}, got '{spatial_mode}'")
        ot_pair_limit = _resolve_ot_block_max_pairs(ot_block_max_pairs, ot_memory_budget_gb)

        if spatial_mode == 'ot_spatial':
            if target_adata is None:
                raise ValueError("target_adata is required when spatial_mode='ot_spatial'.")
            target_coords = target_adata.obsm['spatial']
            n_target = target_coords.shape[0]
            common_genes = reference_adata.var_names.intersection(target_adata.var_names)
            if len(common_genes) < n_genes:
                target_adata = target_adata[:, common_genes].copy()
                reference_matrix = reference_matrix[:, reference_adata.var_names.get_indexer(common_genes)]
                n_genes = len(common_genes)

            source_coords = reference_adata.obsm['spatial']

            # Per-gene max from sparse for boundary / clipping (much cheaper than dense)
            if reference_sparse is not None and len(common_genes) < n_genes:
                clip_ref = reference_sparse[:, reference_adata.var_names.get_indexer(common_genes)]
            else:
                clip_ref = reference_sparse if reference_sparse is not None else reference_matrix

            ot_pair_count = int(n_spots) * int(n_target)
            if ot_pair_count > MAX_DENSE_OT_PAIRS or (
                ot_pair_limit is not None and ot_pair_count > ot_pair_limit
            ):
                transported = _block_ot_transport(
                    reference_matrix,
                    source_coords,
                    target_coords,
                    reg=0.05,
                    block_size=ot_block_size,
                    overlap_frac=ot_overlap_frac,
                    max_block_pairs=ot_pair_limit,
                    transport_rank=False,
                )
                spatial_coords = target_coords.copy()
                n_spots = n_target
                quantile_input = transported
            else:
                cost = cdist(source_coords, target_coords, metric='euclidean')
                a = np.ones(n_spots) / n_spots
                b = np.ones(n_target) / n_target

                from ..de_novo._ot_transport import sinkhorn_transport
                plan = sinkhorn_transport(M=cost, a=a, b=b, reg=0.05)

                col_mass = plan.sum(axis=0, keepdims=True)
                safe_mass = np.where(col_mass > 1e-12, col_mass, 1.0)
                transported = (plan / safe_mass).T @ reference_matrix

                spatial_coords = target_coords.copy()
                n_spots = n_target
                quantile_input = transported
        else:
            spatial_coords = reference_adata.obsm['spatial']
            clip_ref = reference_sparse if reference_sparse is not None else reference_matrix
            quantile_input = reference_matrix

        simulated_matrix = decode_counts_by_spatial_intensity(
            quantile_input,
            model_params,
            boundary_multiplier=boundary_multiplier,
            reference_X=clip_ref,
            random_seed=seed,
        ).astype(np.float32, copy=False)

        boundary_clipped_gene_count = 0
        if clip_overshoot_factor > 0:
            from scipy.sparse import issparse
            if issparse(clip_ref):
                max_ref_counts = np.asarray(clip_ref.max(axis=0).toarray()).ravel()
            else:
                max_ref_counts = np.max(clip_ref, axis=0)
            clip_max = max_ref_counts * (1 + clip_overshoot_factor)
            before = simulated_matrix.copy()
            simulated_matrix = np.clip(simulated_matrix, 0, clip_max)
            boundary_clipped_gene_count = int(np.any(np.abs(before - simulated_matrix) > 1e-12, axis=0).sum())

        obs = target_adata.obs.copy() if spatial_mode == 'ot_spatial' else reference_adata.obs.copy()
        var = target_adata.var.copy() if spatial_mode == 'ot_spatial' else reference_adata.var.copy()
        simulated_adata = ad.AnnData(
            X=simulated_matrix.astype(np.float32),
            obs=obs,
            var=var,
            obsm={'spatial': spatial_coords.copy()}
        )
        simulated_adata.uns['simulation_method'] = 'Quantile_Count_Decoding'
        simulated_adata.uns['simulation_params'] = {
            'clip_overshoot_factor': float(clip_overshoot_factor),
            'boundary_multiplier': float(boundary_multiplier),
            'simulation_mode': mode,
            'spatial_mode': spatial_mode,
            'random_seed': "none" if seed is None else int(seed),
        }
        simulated_adata.uns['simulation_diagnostics'] = _hdf5_safe_metadata(self._build_simulation_diagnostics(
            reference_matrix=reference_matrix,
            simulated_matrix=simulated_matrix,
            model_params=model_params,
            simulation_mode=mode,
            spatial_mode=spatial_mode,
            random_seed=seed,
            boundary_clipped_gene_count=boundary_clipped_gene_count,
            clip_overshoot_factor=clip_overshoot_factor,
        ))
        if model_params.get('parameter_diagnostics', {}).get('requested_config') is not None:
            simulated_adata.uns['alteration_diagnostics'] = _hdf5_safe_metadata({
                'requested_config': model_params['parameter_diagnostics'].get('requested_config'),
                'target_stage_achieved_change': model_params['parameter_diagnostics'].get('target_stage_achieved_change'),
                'realized_stage_achieved_change': simulated_adata.uns['simulation_diagnostics'].get('realized_stage_achieved_change'),
            })
        return simulated_adata

    def _build_simulation_diagnostics(
        self,
        reference_matrix,
        simulated_matrix,
        model_params,
        simulation_mode,
        spatial_mode,
        random_seed,
        boundary_clipped_gene_count,
        clip_overshoot_factor,
    ):
        reference_stats = _gene_stats_from_matrix(reference_matrix, self.reference_adata.var_names)
        realized_stats = _gene_stats_from_matrix(simulated_matrix, self.reference_adata.var_names)
        target_stats = model_params.get('target_stats')
        target_change = None
        if isinstance(target_stats, pd.DataFrame):
            target_change = calculate_fold_change(reference_stats, target_stats)

        parameter_diagnostics = model_params.get('parameter_diagnostics', {})
        diagnostics = {
            'simulation_mode': simulation_mode,
            'gene_parameter_engine': parameter_diagnostics.get('gene_parameter_engine', simulation_mode),
            'assignment_method': parameter_diagnostics.get(
                'assignment_method',
                'identity' if simulation_mode == 'empirical' else 'copula_rank',
            ),
            'spatial_mode': spatial_mode,
            'random_seed': None if random_seed is None else int(random_seed),
            'requested_config': parameter_diagnostics.get('requested_config'),
            'target_fold_change': parameter_diagnostics.get('target_fold_change'),
            'target_stage_achieved_change': target_change or parameter_diagnostics.get('target_stage_achieved_change'),
            'realized_stage_achieved_change': calculate_fold_change(reference_stats, realized_stats),
            'copula_rank_diagnostics': parameter_diagnostics.get('copula_rank_diagnostics', {}),
            'moment_feasibility': parameter_diagnostics.get('moment_feasibility', {'infeasible_gene_count': 0}),
            'boundary_clipping': {
                'clip_overshoot_factor': float(clip_overshoot_factor),
                'clipped_gene_count': int(boundary_clipped_gene_count),
            },
            'model_selection_counts': _model_selection_counts(model_params),
        }
        return diagnostics
    
    def simulate_by_annotation(self, annotation_key: str, **kwargs) -> ad.AnnData:
        """Compatibility path for annotation-key callers."""
        if annotation_key not in self.reference_adata.obs:
            raise KeyError(f"annotation_key '{annotation_key}' not found in adata.obs.")
        fit_kwargs = {
            "visualize_fits": kwargs.get("visualize_fits", False),
            "use_real_stats_directly": kwargs.get("use_real_stats_directly", False),
            "use_heuristic_search": kwargs.get("use_heuristic_search", False),
            "min_accepted_error": kwargs.get("min_accepted_error", 0.5),
            "assignment_weights": kwargs.get("assignment_weights"),
            "screening_pool_size": kwargs.get("screening_pool_size", 100),
            "top_n_to_fully_evaluate": kwargs.get("top_n_to_fully_evaluate", 10),
            "n_jobs": kwargs.get("n_jobs", -1),
            "alteration_config": kwargs.get("alteration_config"),
            "simulation_mode": _translate_parameter_mode(
                kwargs.get("parameter_mode", "hungarian")
            ),
            "spatial_mode": kwargs.get("spatial_mode", "reference_rank"),
            "assignment_method": kwargs.get("assignment_method", "hybrid"),
            "random_seed": kwargs.get("random_seed"),
            "hybrid_alpha": kwargs.get("hybrid_alpha", 0.2),
            "use_distributional_alteration": kwargs.get("use_distributional_alteration", False),
            "ppf_method": kwargs.get("ppf_method", "interp"),
            "beta_n_jobs": kwargs.get("beta_n_jobs", 1),
            "beta_early_stopping_patience": kwargs.get("beta_early_stopping_patience", 2),
            "assignment_solver": kwargs.get("assignment_solver", "scipy"),
            "assignment_n_jobs": kwargs.get("assignment_n_jobs", 1),
            "assignment_blocks": kwargs.get("assignment_blocks", True),
            "assignment_block_size": kwargs.get("assignment_block_size"),
            "assignment_block_multiplier": kwargs.get("assignment_block_multiplier", 8),
            "convert_n_jobs": kwargs.get("convert_n_jobs", 1),
        }
        self.fit_model(**fit_kwargs)
        simulated = self.simulate(
            num_simulation_cores=kwargs.get("num_simulation_cores", 12),
            verbose=kwargs.get("verbose", True),
            clip_overshoot_factor=kwargs.get("clip_overshoot_factor", 0.1),
            boundary_multiplier=kwargs.get("boundary_multiplier", 1.1),
            random_seed=kwargs.get("random_seed"),
            spatial_mode=kwargs.get("spatial_mode", "reference_rank"),
            target_adata=kwargs.get("target_adata"),
            ot_block_size=kwargs.get("ot_block_size", 40000),
            ot_overlap_frac=kwargs.get("ot_overlap_frac", 0.25),
            ot_block_max_pairs=kwargs.get("ot_block_max_pairs"),
            ot_memory_budget_gb=kwargs.get("ot_memory_budget_gb"),
        )
        simulated.uns["annotation_key"] = annotation_key
        return simulated
    

def _block_ot_transport(
    reference_matrix,
    source_coords,
    target_coords,
    reg=0.05,
    block_size=40000,
    overlap_frac=0.25,
    max_block_pairs=None,
    memory_budget_gb=None,
    transport_rank=True,
):
    """Block-based optimal transport for large datasets.

    Partitions target space into grid tiles, computes OT per tile with
    overlap, and assembles results.  Uncovered spots fall back to
    nearest-neighbour assignment.  Avoids the O(n^2) dense cost matrix
    that would OOM for Xenium-scale (> 100k spots) datasets.

    ``max_block_pairs`` and ``memory_budget_gb`` are opt-in controls.  When
    omitted, each grid tile is processed exactly as before.  When provided,
    oversized tiles are recursively split before dense cost/Sinkhorn arrays
    are allocated.
    """
    from ..de_novo._ot_transport import sinkhorn_transport
    from ..de_novo.quantile_field import midpoint_rank_normalize

    n_source = source_coords.shape[0]
    n_target = target_coords.shape[0]
    block_size = int(block_size)
    overlap_frac = float(overlap_frac)
    if block_size < 1:
        raise ValueError("ot_block_size must be a positive integer.")
    if not np.isfinite(overlap_frac) or overlap_frac < 0:
        raise ValueError("ot_overlap_frac must be a non-negative finite number.")
    max_block_pairs = _resolve_ot_block_max_pairs(max_block_pairs, memory_budget_gb)

    if transport_rank:
        source_values = midpoint_rank_normalize(reference_matrix, tie_policy='stable_ordinal', clip_eps=1e-6)
    else:
        source_values = np.asarray(reference_matrix, dtype=np.float64)

    tgt_x, tgt_y = target_coords[:, 0], target_coords[:, 1]
    src_x, src_y = source_coords[:, 0], source_coords[:, 1]

    x_min, x_max = tgt_x.min(), tgt_x.max()
    y_min, y_max = tgt_y.min(), tgt_y.max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    n_tiles = max(1, int(np.ceil(np.sqrt(n_target / max(1, block_size)))))
    x_edges = np.linspace(x_min, x_max, n_tiles + 1)
    y_edges = np.linspace(y_min, y_max, n_tiles + 1)
    x_overlap = overlap_frac * (x_range / n_tiles)
    y_overlap = overlap_frac * (y_range / n_tiles)

    transported_accum = np.zeros((n_target, reference_matrix.shape[1]), dtype=np.float64)
    count_accum = np.zeros(n_target, dtype=np.float64)

    for ix in range(n_tiles):
        for iy in range(n_tiles):
            xl = max(x_min, x_edges[ix] - x_overlap)
            xr = min(x_max, x_edges[ix + 1] + x_overlap)
            yl = max(y_min, y_edges[iy] - y_overlap)
            yr = min(y_max, y_edges[iy + 1] + y_overlap)

            tgt_mask = (tgt_x >= xl) & (tgt_x <= xr) & (tgt_y >= yl) & (tgt_y <= yr)
            tgt_idx = np.where(tgt_mask)[0]
            if len(tgt_idx) < 5:
                continue

            src_mask = (src_x >= xl) & (src_x <= xr) & (src_y >= yl) & (src_y <= yr)
            src_idx = np.where(src_mask)[0]
            if len(src_idx) < 5:
                continue

            for sub_src_idx, sub_tgt_idx in _iter_bounded_ot_subblocks(
                src_idx,
                tgt_idx,
                source_coords,
                target_coords,
                max_block_pairs,
            ):
                if len(sub_tgt_idx) < 5 or len(sub_src_idx) < 5:
                    continue

                cost = cdist(
                    source_coords[sub_src_idx],
                    target_coords[sub_tgt_idx],
                    metric='euclidean',
                )
                a = np.ones(len(sub_src_idx)) / n_source
                b = np.ones(len(sub_tgt_idx)) / n_target

                plan = sinkhorn_transport(M=cost, a=a, b=b, reg=reg)

                col_mass = plan.sum(axis=0, keepdims=True)
                safe_mass = np.where(col_mass > 1e-12, col_mass, 1.0)
                transported_local = (plan / safe_mass).T @ source_values[sub_src_idx, :]

                transported_accum[sub_tgt_idx, :] += transported_local
                count_accum[sub_tgt_idx] += 1

    uncovered = count_accum == 0
    if np.any(uncovered):
        uncovered_idx = np.where(uncovered)[0]
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(source_coords)
        nearest_src = nn.kneighbors(target_coords[uncovered_idx], return_distance=False).ravel()
        transported_accum[uncovered_idx, :] = source_values[nearest_src, :]
        count_accum[uncovered_idx] = 1.0

    transported = transported_accum / count_accum[:, None]

    if transport_rank:
        transported = midpoint_rank_normalize(transported, tie_policy='stable_ordinal', clip_eps=1e-6)
    return transported


def simulate_single_slice(adata: ad.AnnData, visualize_fits: bool = False, num_simulation_cores: int = 12, verbose: bool = True, clip_overshoot_factor: float = 0.1, use_real_stats_directly: bool = False, annotation_key: str = None, use_heuristic_search: bool = False, min_accepted_error: float = 0.005, assignment_weights: dict = None, screening_pool_size: int = 1000, top_n_to_fully_evaluate: int = 10, n_jobs: int = -1, alteration_config=None, boundary_multiplier: float = 1.1, parameter_mode: str = 'hungarian', spatial_mode: str = 'reference_rank', target_adata=None, assignment_method: str = 'hybrid', random_seed: int = None, hybrid_alpha: float = 0.2, use_distributional_alteration: bool = False, ppf_method: str = 'interp', beta_n_jobs: int = 1, beta_early_stopping_patience: int = 2, assignment_solver: str = 'scipy', assignment_n_jobs: int = 1, assignment_blocks: bool = True, assignment_block_size: int = None, assignment_block_multiplier: int = 8, convert_n_jobs: int = 1, ot_block_size: int = 40000, ot_overlap_frac: float = 0.25, ot_block_max_pairs: int = None, ot_memory_budget_gb: float = None) -> ad.AnnData:
    """
    Run single-slice simulation.

    Args:
        boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max).
        alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions.
        parameter_mode: 'hungarian' (copula + Hungarian assignment) or 'reference_stats' (direct reference stats).
        spatial_mode: 'reference_rank' (rank-preserving) or 'ot_spatial' (OT transport, requires target_adata).
        target_adata: Target AnnData for ot_spatial mode (required when spatial_mode='ot_spatial').
        assignment_method: 'hybrid' or 'copula_rank' — only meaningful when parameter_mode='hungarian'.
        random_seed: Optional seed for reproducible generative sampling.
        use_distributional_alteration: If True, alter marginal model parameters (θ → θ').
        ppf_method: 'interp' (fast, default) or 'exact' (bit-identical fsolve).
        beta_n_jobs: parallel workers for Beta component search (1=sequential).
        beta_early_stopping_patience: patience for Beta mixture early stopping (0 = no early stop).
        assignment_blocks: use batched Hungarian (True, default) or full global (False).
        assignment_block_multiplier: candidates per gene in each batch (default 8).
        convert_n_jobs: parallel workers for count-model parameter conversion (1=sequential).
        ot_block_size: Target spots per OT spatial grid tile.
        ot_overlap_frac: Fractional spatial overlap between OT grid tiles.
        ot_block_max_pairs: Optional hard cap for dense source-target pairs
            in each OT sub-block. Defaults to None, preserving existing block
            behavior.
        ot_memory_budget_gb: Optional memory budget converted to an OT
            sub-block pair cap. Defaults to None.
    """
    simulation_mode = _translate_parameter_mode(parameter_mode)
    spatial_mode = _translate_spatial_mode(spatial_mode)
    if spatial_mode == "ot_spatial" and target_adata is None:
        raise ValueError("target_adata is required when spatial_mode='ot_spatial'.")

    if simulation_mode == 'empirical' and assignment_method not in (None, 'hybrid', 'identity'):
        warnings.warn(
            f"assignment_method='{assignment_method}' is ignored when "
            f"parameter_mode='reference_stats'. Assignment is always 'identity'.",
            UserWarning,
            stacklevel=2,
        )
    if verbose: print("Starting comprehensive single slice simulation...")
    adata = adata.copy()
    safe_calculate_qc_metrics(adata, verbose=verbose)
    simulator = SpatialSimulator(adata)
    
    heuristic_kwargs = {
        'use_heuristic_search': use_heuristic_search,
        'min_accepted_error': min_accepted_error,
        'assignment_weights': assignment_weights,
        'screening_pool_size': screening_pool_size,
        'top_n_to_fully_evaluate': top_n_to_fully_evaluate,
        'n_jobs': n_jobs,
        'simulation_mode': simulation_mode,
        'spatial_mode': spatial_mode,
        'assignment_method': assignment_method,
        'random_seed': random_seed,
        'hybrid_alpha': hybrid_alpha,
        'use_distributional_alteration': use_distributional_alteration,
        'ppf_method': ppf_method,
        'beta_n_jobs': beta_n_jobs,
        'beta_early_stopping_patience': beta_early_stopping_patience,
        'assignment_solver': assignment_solver,
        'assignment_n_jobs': assignment_n_jobs,
        'assignment_blocks': assignment_blocks,
        'assignment_block_size': assignment_block_size,
        'assignment_block_multiplier': assignment_block_multiplier,
        'convert_n_jobs': convert_n_jobs,
    }

    if annotation_key:
        if use_real_stats_directly: print("Warning: `use_real_stats_directly` is not implemented for annotation-based simulation. Running standard simulation.")
        if verbose: print(f"Using annotation-based simulation with key: '{annotation_key}'")
        simulated_adata = simulator.simulate_by_annotation(
            annotation_key=annotation_key,
            visualize_fits=visualize_fits,
            num_simulation_cores=num_simulation_cores,
            verbose=verbose,
            clip_overshoot_factor=clip_overshoot_factor,
            boundary_multiplier=boundary_multiplier,
            alteration_config=alteration_config,
            target_adata=target_adata,
            ot_block_size=ot_block_size,
            ot_overlap_frac=ot_overlap_frac,
            ot_block_max_pairs=ot_block_max_pairs,
            ot_memory_budget_gb=ot_memory_budget_gb,
            **heuristic_kwargs, # Pass all heuristic controls
        )


    else:
        if use_real_stats_directly:
            if verbose: print("--- RUNNING IN DIAGNOSTIC MODE (USING REAL STATS) ---")
        elif use_heuristic_search:
            if verbose: print("--- RUNNING IN BOOSTED HEURISTIC OPTIMIZATION MODE ---")
        else:
            if verbose:
                print("--- RUNNING IN STANDARD DETERMINISTIC MODE ---")
        
        simulator.fit_model(
            visualize_fits=visualize_fits, 
            use_real_stats_directly=use_real_stats_directly,
            alteration_config=alteration_config,  # Pass alteration configuration
            **heuristic_kwargs # Pass all heuristic controls
        )
        simulated_adata = simulator.simulate(
            num_simulation_cores=num_simulation_cores,
            verbose=verbose,
            clip_overshoot_factor=clip_overshoot_factor,
            boundary_multiplier=boundary_multiplier,
            random_seed=random_seed,
            spatial_mode=spatial_mode,
            target_adata=target_adata,
            ot_block_size=ot_block_size,
            ot_overlap_frac=ot_overlap_frac,
            ot_block_max_pairs=ot_block_max_pairs,
            ot_memory_budget_gb=ot_memory_budget_gb,
        )
        
    if verbose: print(f"\nSimulation completed successfully!")
    return simulated_adata
