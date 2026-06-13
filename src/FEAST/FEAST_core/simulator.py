import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import warnings

from .count_decoding import decode_counts_from_quantiles
from .parameter_cloud import (
    STAT_COLUMNS,
    GeneParameterSimulator,
    alteration_config_to_dict,
    calculate_fold_change,
    convert_params_for_new_simulator,
    resolve_simulation_mode,
)

QUANTILE_CALIBRATION_SOURCES = ("reference_rank", "raw")


def resolve_quantile_calibration_source(quantile_calibration=None, simulation_mode: str = "generative") -> str:
    """Normalize the source of spot-gene quantiles used during decoding."""
    mode = resolve_simulation_mode(simulation_mode)
    if quantile_calibration is None:
        return "reference_rank" if mode == "empirical" else "raw"
    source = str(quantile_calibration).lower().strip()
    if source in {"auto", "default"}:
        return "reference_rank" if mode == "empirical" else "raw"
    aliases = {
        "rank": "reference_rank",
        "reference": "reference_rank",
        "reference_rank": "reference_rank",
        "empirical": "reference_rank",
        "empirical_rank": "reference_rank",
        "iid": "raw",
        "uniform": "raw",
        "raw": "raw",
    }
    source = aliases.get(source, source)
    if source not in QUANTILE_CALIBRATION_SOURCES:
        raise ValueError("quantile_calibration must be 'reference_rank', 'raw', or 'auto'.")
    return source


def safe_calculate_qc_metrics(adata, verbose=False):
    try:
        if adata.n_vars > 0 and adata.n_obs > 0: sc.pp.calculate_qc_metrics(adata, percent_top=[20, 50, 100] if adata.n_vars > 100 else [50], inplace=True, log1p=False)
    except Exception as e:
        if verbose: print(f"Warning: QC calculation failed ({e}), using basic metrics only")
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
    matrix = np.asarray(matrix, dtype=np.float64)
    n_obs = matrix.shape[0]
    return pd.DataFrame({
        'mean': np.mean(matrix, axis=0),
        'variance': np.var(matrix, axis=0),
        'zero_prop': 1 - (np.count_nonzero(matrix, axis=0) / n_obs),
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


def run_parameter_cloud_fitting(adata, visualize_fits=False, use_heuristic_search=True, min_accepted_error=0.5, assignment_weights=None, screening_pool_size=100, top_n_to_fully_evaluate=10, n_jobs=-1, alteration_config=None, simulation_mode='generative', quantile_calibration=None, random_seed=None, hybrid_alpha=0.2):
    """
    Build an integrated FEAST gene-parameter table and convert it to count-model parameters.

    Args:
        alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions
        hybrid_alpha (float): Weight for log-space distance in hybrid OT cost (0.2 = 20% log, 80% raw).
                              Set to 1.0 for old pure-log-space behavior.
    """
    print("\n>>> Entering STANDARD fitting pipeline: parameter_cloud <<<")

    if assignment_weights is None:
        assignment_weights = {'mean': 3, 'variance': 1, 'zero_prop': 1.0}
    mode = resolve_simulation_mode(simulation_mode)
    quantile_source = resolve_quantile_calibration_source(quantile_calibration, mode)
    if use_heuristic_search:
        warnings.warn(
            "use_heuristic_search is retained for compatibility but is ignored by the "
            "integrated simulation_mode pipeline; generative mode uses Copula-rank OT.",
            RuntimeWarning,
            stacklevel=2,
        )

    simulator = GeneParameterSimulator()
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
        verbose=True,
    )

    model_params = convert_params_for_new_simulator(
        assigned_synthetic_params, n_spots=adata.n_obs)
    model_params['simulation_evaluation'] = {
        'source': 'integrated_parameter_cloud',
        'simulation_mode': mode,
        'quantile_calibration': quantile_source,
    }
    model_params['simulation_mode'] = mode
    model_params['random_seed'] = None if random_seed is None else int(random_seed)
    model_params['target_stats'] = assigned_synthetic_params
    model_params['parameter_diagnostics'] = diagnostics
    model_params['count_decode_method'] = 'quantile'
    model_params['quantile_calibration'] = quantile_source
    print(">>> Exiting parameter_cloud pipeline <<<\n")
    return model_params

def run_direct_fitting_from_real_stats(adata):
    """This diagnostic pipeline remains unchanged."""
    print("\n>>> Entering DIAGNOSTIC fitting pipeline: Using REAL stats directly <<<")
    simulator = GeneParameterSimulator()
    simulator.fit_statistics_only(adata)
    real_stats_for_conversion = simulator.original_stats.reset_index().rename(columns={'index': 'gene_id'})
    model_params = convert_params_for_new_simulator(
        real_stats_for_conversion, n_spots=adata.n_obs)
    model_params['simulation_evaluation'] = {'source': 'direct_from_real_stats', 'simulation_mode': 'empirical'}
    model_params['simulation_mode'] = 'empirical'
    model_params['target_stats'] = real_stats_for_conversion
    model_params['count_decode_method'] = 'quantile'
    model_params['quantile_calibration'] = 'reference_rank'
    print(">>> Diagnostic fitting complete <<<\n")
    return model_params


class SpatialSimulator:
    def __init__(self, reference_adata: ad.AnnData, model_params: dict = None):
        if 'spatial' not in reference_adata.obsm: raise ValueError("Reference AnnData must contain 'spatial' coordinates.")
        self.reference_adata = reference_adata.copy() 
        self.reference_adata.var_names_make_unique()
        self.reference_adata.obs_names_make_unique()
        self._model_params = model_params

    def fit_model(self, visualize_fits: bool = False, use_real_stats_directly: bool = False, use_heuristic_search: bool = False, min_accepted_error: float = 0.5, assignment_weights: dict = None, screening_pool_size: int = 100, top_n_to_fully_evaluate: int = 10, n_jobs: int = -1, alteration_config=None, simulation_mode: str = 'generative', quantile_calibration=None, random_seed: int = None, hybrid_alpha: float = 0.2) -> 'SpatialSimulator':
        """
        UPDATED: Exposes the new boosted heuristic search parameters and marginal distribution alteration.
        
        Args:
            alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions
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
                quantile_calibration=quantile_calibration,
                random_seed=random_seed,
                hybrid_alpha=hybrid_alpha,
            )
        return self
    
    def set_model_params(self, model_params: dict):
        """Set model parameters directly."""
        self._model_params = model_params
        return self
    
    def get_model_params(self):
        """Get current model parameters."""
        return self._model_params
    
    def simulate(self, num_simulation_cores: int = 12, verbose: bool = True, clip_overshoot_factor: float = 0.0, boundary_multiplier: float = 1.1, random_seed: int = None) -> ad.AnnData:
        """
        Args:
            num_simulation_cores (int): Number of cores for simulation (legacy parameter).
            verbose (bool): If True, prints progress updates.
            clip_overshoot_factor (float): Factor to clip max expression values relative to reference.
            boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max).
            random_seed (int, optional): Seed for raw generative quantile decoding.
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
        )
        
        if verbose:
            print(f"Quantile simulation complete for {simulated_adata.n_obs} spots and {simulated_adata.n_vars} genes")
        
        safe_calculate_qc_metrics(simulated_adata)
        return simulated_adata

    def _apply_quantile_count_decoding(self, reference_adata, model_params, verbose=True, clip_overshoot_factor=0.0, boundary_multiplier=1.1, random_seed=None):
        """Generate counts from model parameters through the integrated quantile decoder."""
        reference_matrix = _dense_matrix(reference_adata.X, dtype=np.float64)
        spatial_coords = reference_adata.obsm['spatial']
        n_spots, n_genes = reference_matrix.shape

        mode = resolve_simulation_mode(model_params.get('simulation_mode', 'empirical'))
        diagnostics_seed = model_params.get('random_seed', None)
        seed = random_seed if random_seed is not None else diagnostics_seed
        quantile_calibration = model_params.get(
            'quantile_calibration',
            'reference_rank' if mode == 'empirical' else 'raw',
        )

        if quantile_calibration == 'reference_rank':
            quantile_input = reference_matrix
            decoder_calibration = 'rank'
        elif quantile_calibration == 'raw':
            rng = np.random.default_rng(None if seed is None else int(seed))
            quantile_input = rng.random((n_spots, n_genes), dtype=np.float64)
            decoder_calibration = 'raw'
        else:
            raise ValueError("quantile_calibration must be 'reference_rank' or 'raw' for integrated simulation.")

        simulated_matrix = decode_counts_from_quantiles(
            quantile_input,
            model_params,
            method='quantile',
            quantile_calibration=decoder_calibration,
            boundary_multiplier=boundary_multiplier,
            reference_X=reference_matrix,
            random_seed=seed,
        ).astype(np.float32, copy=False)

        boundary_clipped_gene_count = 0
        if clip_overshoot_factor > 0:
            max_ref_counts = np.max(reference_matrix, axis=0)
            clip_max = max_ref_counts * (1 + clip_overshoot_factor)
            before = simulated_matrix.copy()
            simulated_matrix = np.clip(simulated_matrix, 0, clip_max)
            boundary_clipped_gene_count = int(np.any(np.abs(before - simulated_matrix) > 1e-12, axis=0).sum())

        simulated_adata = ad.AnnData(
            X=simulated_matrix.astype(np.float32),
            obs=reference_adata.obs.copy(),
            var=reference_adata.var.copy(),
            obsm={'spatial': spatial_coords.copy()}
        )
        simulated_adata.uns['simulation_method'] = 'Quantile_Count_Decoding'
        simulated_adata.uns['simulation_params'] = {
            'clip_overshoot_factor': float(clip_overshoot_factor),
            'boundary_multiplier': float(boundary_multiplier),
            'simulation_mode': mode,
            'random_seed': "none" if seed is None else int(seed),
        }
        simulated_adata.uns['simulation_diagnostics'] = _hdf5_safe_metadata(self._build_simulation_diagnostics(
            reference_matrix=reference_matrix,
            simulated_matrix=simulated_matrix,
            model_params=model_params,
            simulation_mode=mode,
            quantile_calibration=quantile_calibration,
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
        quantile_calibration,
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
                'identity' if simulation_mode == 'empirical' else 'copula_rank_ot',
            ),
            'count_decode_method': 'quantile',
            'quantile_calibration': quantile_calibration,
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
    
    def _apply_deterministic_rank_assignment(self, reference_adata, model_params, verbose=True, clip_overshoot_factor=0.0, boundary_multiplier=1.1, n_modules=30, n_neighbors=6):
        """Generate counts from model parameters while preserving per-gene reference ranks."""
        reference_matrix = reference_adata.X.toarray() if hasattr(reference_adata.X, 'toarray') else reference_adata.X.copy()
        spatial_coords = reference_adata.obsm['spatial']

        if verbose:
            print("Applying deterministic rank-preserving assignment...")

        new_counts = self._generate_counts_from_parameters(reference_adata, model_params, verbose, boundary_multiplier)
        simulated_matrix = np.zeros_like(reference_matrix, dtype=np.float32)

        for gene_idx in range(reference_matrix.shape[1]):
            original_spatial_ranks = np.argsort(reference_matrix[:, gene_idx], kind="mergesort")
            new_values_sorted = np.sort(new_counts[:, gene_idx])
            simulated_matrix[original_spatial_ranks, gene_idx] = new_values_sorted
        
        if clip_overshoot_factor > 0:
            max_ref_counts = np.max(reference_matrix, axis=0)
            clip_max = max_ref_counts * (1 + clip_overshoot_factor)
            simulated_matrix = np.clip(simulated_matrix, 0, clip_max)
        
        simulated_adata = ad.AnnData(
            X=simulated_matrix.astype(np.float32),
            obs=reference_adata.obs.copy(),
            var=reference_adata.var.copy(),
            obsm={'spatial': spatial_coords.copy()}
        )
        simulated_adata.uns['simulation_method'] = 'Deterministic_Rank_Preservation'
        simulated_adata.uns['simulation_params'] = {
            'clip_overshoot_factor': clip_overshoot_factor
        }
        return simulated_adata
    
    def _find_gene_modules(self, new_counts, n_modules, verbose=False):
        """Find co-expression modules and their leader genes using NMF."""
        from sklearn.decomposition import NMF
        
        # Ensure we don't request more modules than genes
        n_modules = min(n_modules, new_counts.shape[1] // 2)
        
        if verbose:
            print(f"Running NMF with {n_modules} components...")
        
        # NMF to find additive parts-based representations
        model = NMF(n_components=n_modules, init='random', random_state=42, max_iter=500)
        try:
            W = model.fit_transform(new_counts)  # spot loadings for each module
            H = model.components_  # gene loadings for each module
        except Exception as e:
            if verbose:
                print(f"NMF failed ({e}), using simple clustering fallback")
            return self._simple_gene_clustering(new_counts, n_modules)
        
        # For each module, find the gene with highest loading (the "leader")
        leader_genes_indices = np.argmax(H, axis=1)
        
        # Assign each gene to the module where it has highest loading
        gene_modules = [[] for _ in range(n_modules)]
        gene_to_module_map = np.argmax(H, axis=0)
        
        for gene_idx, module_idx in enumerate(gene_to_module_map):
            gene_modules[module_idx].append(gene_idx)
        
        if verbose:
            module_sizes = [len(module) for module in gene_modules]
            print(f"Created modules with sizes: {module_sizes}")
        
        return gene_modules, leader_genes_indices
    
    def _simple_gene_clustering(self, new_counts, n_modules):
        """Fallback clustering method if NMF fails."""
        n_genes = new_counts.shape[1]
        genes_per_module = n_genes // n_modules
        
        gene_modules = []
        leader_genes_indices = []
        
        for i in range(n_modules):
            start_idx = i * genes_per_module
            end_idx = start_idx + genes_per_module if i < n_modules - 1 else n_genes
            module_genes = list(range(start_idx, end_idx))
            gene_modules.append(module_genes)
            leader_genes_indices.append(start_idx)  # First gene as leader
        
        return gene_modules, np.array(leader_genes_indices)
    
    def _create_spatial_smoother(self, spatial_coords, n_neighbors):
        """Create a normalized adjacency matrix for spatial smoothing."""
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse import csgraph
        
        # Build k-NN graph
        adj_matrix = kneighbors_graph(
            spatial_coords, 
            n_neighbors=min(n_neighbors, len(spatial_coords)-1), 
            mode='connectivity', 
            include_self=True
        )
        
        # Normalize to create smoothing operator
        smoother = adj_matrix / adj_matrix.sum(axis=1)
        
        return smoother
    
    def _guided_assignment_core(self, reference_matrix, new_counts, spatial_smoother,
                               gene_modules, leader_genes_indices, verbose=False):
        """Compatibility wrapper for deterministic rank assignment."""
        n_spots, n_genes = reference_matrix.shape
        S_final = np.zeros_like(reference_matrix, dtype=np.float32)
        reference_ranks = np.argsort(reference_matrix, axis=0)
        new_counts_ranks = np.argsort(new_counts, axis=0)
        shuffled_indices = np.arange(n_spots, dtype=int)
        for gene_idx in range(n_genes):
            assigned_indices = new_counts_ranks[shuffled_indices, gene_idx]
            S_final[reference_ranks[:, gene_idx], gene_idx] = \
                new_counts[assigned_indices, gene_idx]
        return S_final
    
    def _resolve_assignment_conflicts(self, assignment_map):
        """
        Resolve conflicts when multiple ranks are assigned to the same spot.
        Use a greedy approach to ensure each spot gets exactly one rank.
        """
        n_spots = len(assignment_map)
        resolved_map = np.zeros(n_spots, dtype=int)
        used_spots = set()
        
        # First pass: assign non-conflicting mappings
        conflicts = []
        for rank in range(n_spots):
            target_spot = assignment_map[rank]
            if target_spot not in used_spots:
                resolved_map[rank] = target_spot
                used_spots.add(target_spot)
            else:
                conflicts.append(rank)
        
        # Second pass: resolve conflicts by assigning to unused spots
        available_spots = [i for i in range(n_spots) if i not in used_spots]
        
        for i, rank in enumerate(conflicts):
            if i < len(available_spots):
                resolved_map[rank] = available_spots[i]
            else:
                # Fallback: assign to any remaining spot (shouldn't happen with proper implementation)
                resolved_map[rank] = rank
        
        return resolved_map
    
    def _generate_counts_from_parameters(self, reference_adata, model_params, verbose=False, boundary_multiplier=1.1):
        """Generate new count matrix from fitted statistical distribution parameters.
        
        Args:
            boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max)
        """
        if verbose:
            print("Generating new counts from fitted statistical distributions...")
        
        # Extract the fitted distribution parameters
        if 'genes' not in model_params or 'model_selected' not in model_params or 'marginal_param1' not in model_params:
            if verbose:
                print("Warning: model_params missing required keys, falling back to reference-based simulation")
            return self._fallback_reference_based_simulation(reference_adata, boundary_multiplier)
        
        n_spots, n_genes = reference_adata.shape
        new_counts = np.zeros((n_spots, n_genes), dtype=np.float32)
        
        # Calculate maximum counts per gene from reference data for boundary constraint
        reference_matrix = reference_adata.X.toarray() if hasattr(reference_adata.X, 'toarray') else reference_adata.X
        max_counts_per_gene = np.max(reference_matrix, axis=0)
        # Set boundary using the tunable multiplier
        boundary_per_gene = max_counts_per_gene * boundary_multiplier
        
        if verbose:
            print(f"Applying {boundary_multiplier*100:.0f}% boundary constraint based on reference max counts")
            print(f"Max reference counts range: [{np.min(max_counts_per_gene):.1f}, {np.max(max_counts_per_gene):.1f}]")
            print(f"Boundary range: [{np.min(boundary_per_gene):.1f}, {np.max(boundary_per_gene):.1f}]")
        
        # Sample from fitted distributions for each gene
        for gene_idx in range(n_genes):
            if gene_idx >= len(model_params['model_selected']) or gene_idx >= len(model_params['marginal_param1']):
                if verbose:
                    print(f"Warning: No parameters for gene {gene_idx}, using reference values")
                if hasattr(reference_adata.X, 'toarray'):
                    new_counts[:, gene_idx] = reference_adata.X[:, gene_idx].toarray().flatten()
                else:
                    new_counts[:, gene_idx] = reference_adata.X[:, gene_idx]
                continue
            
            model_type = model_params['model_selected'][gene_idx]
            params = model_params['marginal_param1'][gene_idx]  # [pi0, r, mean_param]
            
            try:
                # Ensure params has enough elements - pad with defaults if needed
                if not isinstance(params, (list, tuple, np.ndarray)) or len(params) < 3:
                    # Pad with safe defaults: [pi0=0.1, r=1.0, mean_param=1.0]
                    params_safe = [0.1, 1.0, 1.0]
                    if isinstance(params, (list, tuple, np.ndarray)):
                        for i in range(min(len(params), 3)):
                            if i < len(params) and np.isfinite(params[i]):
                                params_safe[i] = params[i]
                    params = params_safe
                    if verbose:
                        print(f"Warning: Gene {gene_idx} has insufficient parameters ({len(model_params['marginal_param1'][gene_idx]) if isinstance(model_params['marginal_param1'][gene_idx], (list, tuple, np.ndarray)) else 0}), using defaults")
                
                # Safe parameter extraction with bounds checking
                def safe_param(idx, default_val):
                    try:
                        if idx < len(params) and np.isfinite(params[idx]):
                            return max(params[idx], 1e-8) if idx > 0 else np.clip(params[idx], 0, 1) if idx == 0 else params[idx]
                        return default_val
                    except (IndexError, TypeError):
                        return default_val
                
                # Sample from the appropriate distribution using safe parameter extraction
                if model_type == 'Poisson':
                    lambda_param = safe_param(2, 1.0)  # mean_param
                    gene_counts = np.random.poisson(lambda_param, size=n_spots)
                
                elif model_type == 'NB':  # Negative Binomial
                    mu = safe_param(2, 1.0)  # mean_param
                    r = safe_param(1, 1000.0)  # dispersion
                    
                    # Convert to n, p parameterization for numpy with validation
                    if mu <= 0 or r <= 0:
                        # Fallback to Poisson if NB parameters are invalid
                        gene_counts = np.random.poisson(max(mu, 1e-8), size=n_spots)
                    else:
                        p = r / (r + mu)
                        n = r
                        
                        # Validate NB parameters
                        if not (0 < p <= 1 and n > 0):
                            # Fallback to Poisson if parameters are still invalid
                            gene_counts = np.random.poisson(mu, size=n_spots)
                        else:
                            gene_counts = np.random.negative_binomial(n, p, size=n_spots)
                
                elif model_type == 'ZIP':  # Zero-Inflated Poisson
                    pi0 = safe_param(0, 0.1)  # zero inflation probability
                    lambda_param = safe_param(2, 1.0)  # mean_param
                    
                    # Sample zero inflation
                    zero_mask = np.random.binomial(1, pi0, size=n_spots).astype(bool)
                    gene_counts = np.random.poisson(lambda_param, size=n_spots)
                    gene_counts[zero_mask] = 0
                
                elif model_type == 'ZINB':  # Zero-Inflated Negative Binomial
                    pi0 = safe_param(0, 0.1)  # zero inflation probability
                    mu = safe_param(2, 1.0)  # mean_param  
                    r = safe_param(1, 1000.0)  # dispersion
                    
                    # Convert to n, p parameterization for numpy with validation
                    if mu <= 0 or r <= 0:
                        # Fallback to Poisson if NB parameters are invalid
                        gene_counts = np.random.poisson(max(mu, 1e-8), size=n_spots)
                    else:
                        p = r / (r + mu)
                        n = r
                        
                        # Validate NB parameters
                        if not (0 < p <= 1 and n > 0):
                            # Fallback to Poisson if parameters are still invalid
                            gene_counts = np.random.poisson(mu, size=n_spots)
                        else:
                            # Sample zero inflation
                            zero_mask = np.random.binomial(1, pi0, size=n_spots).astype(bool)
                            gene_counts = np.random.negative_binomial(n, p, size=n_spots)
                            gene_counts[zero_mask] = 0
                
                else:
                    if verbose:
                        print(f"Warning: Unknown model type '{model_type}' for gene {gene_idx}, using Poisson fallback")
                    lambda_param = safe_param(2, 1.0)
                    gene_counts = np.random.poisson(lambda_param, size=n_spots)
                
                new_counts[:, gene_idx] = gene_counts.astype(np.float32)
                
                # Apply boundary constraint: resample until all values within boundary
                gene_boundary = boundary_per_gene[gene_idx]
                violations_mask = new_counts[:, gene_idx] > gene_boundary
                n_violations = np.sum(violations_mask)
                
                if n_violations > 0:
                    violation_indices = np.where(violations_mask)[0]
                    n_resampled = 0
                    max_resample_attempts = 100  # Prevent infinite loops
                    
                    # Resample violations using the same distribution until all within boundary
                    for attempt in range(max_resample_attempts):
                        if n_violations == 0:
                            break
                            
                        # Resample based on the fitted distribution using safe parameter access
                        if model_type == 'Poisson':
                            lambda_param = safe_param(2, 1.0)
                            resampled_values = np.random.poisson(lambda_param, size=n_violations)
                            
                        elif model_type == 'NB':
                            mu = safe_param(2, 1.0)
                            alpha = safe_param(1, 1000.0)  # Use r instead of alpha for consistency
                            if mu <= 0 or alpha <= 0:
                                resampled_values = np.random.poisson(max(mu, 1e-8), size=n_violations)
                            else:
                                p = alpha / (alpha + mu)
                                n = alpha
                                p = np.clip(p, 1e-8, 1-1e-8)
                                resampled_values = np.random.negative_binomial(n, p, size=n_violations)
                            
                        elif model_type == 'ZIP':
                            pi0 = safe_param(0, 0.1)
                            lambda_param = safe_param(2, 1.0)
                            zero_mask = np.random.random(n_violations) < pi0
                            resampled_values = np.random.poisson(lambda_param, size=n_violations)
                            resampled_values[zero_mask] = 0
                            
                        elif model_type == 'ZINB':
                            pi0 = safe_param(0, 0.1)
                            mu = safe_param(2, 1.0)
                            alpha = safe_param(1, 1000.0)  # Use r instead of alpha for consistency
                            if mu <= 0 or alpha <= 0:
                                resampled_values = np.random.poisson(max(mu, 1e-8), size=n_violations)
                            else:
                                p = alpha / (alpha + mu)
                                n = alpha
                                p = np.clip(p, 1e-8, 1-1e-8)
                                zero_mask = np.random.random(n_violations) < pi0
                                resampled_values = np.random.negative_binomial(n, p, size=n_violations)
                                resampled_values[zero_mask] = 0
                            
                        else:
                            # Fallback to Poisson
                            lambda_param = safe_param(2, 1.0)
                            resampled_values = np.random.poisson(lambda_param, size=n_violations)
                        
                        # Only keep values within boundary
                        valid_mask = resampled_values <= gene_boundary
                        valid_values = resampled_values[valid_mask]
                        n_valid = len(valid_values)
                        
                        if n_valid > 0:
                            # Replace the first n_valid violations with valid resampled values
                            update_indices = violation_indices[:n_valid]
                            new_counts[update_indices, gene_idx] = valid_values.astype(np.float32)
                            n_resampled += n_valid
                            
                            # Update violation tracking
                            violation_indices = violation_indices[n_valid:]
                            n_violations = len(violation_indices)
                    
                    # If still have violations after max attempts, use truncated uniform sampling
                    if n_violations > 0:
                        # Sample uniformly within [0, gene_boundary] for remaining violations
                        uniform_values = np.random.uniform(0, gene_boundary, size=n_violations)
                        new_counts[violation_indices, gene_idx] = uniform_values.astype(np.float32)
                        n_resampled += n_violations
                    
                    if verbose and n_resampled > 0:
                        print(f"  Gene {gene_idx}: Resampled {n_resampled} values to respect boundary {gene_boundary:.1f}")
                
            except Exception as e:
                if verbose:
                    print(f"Warning: Sampling failed for gene {gene_idx} with model {model_type}: {e}")
                # Fallback to reference values
                if hasattr(reference_adata.X, 'toarray'):
                    new_counts[:, gene_idx] = reference_adata.X[:, gene_idx].toarray().flatten()
                else:
                    new_counts[:, gene_idx] = reference_adata.X[:, gene_idx]
        
        if verbose:
            print(f"Generated counts from distributions: Poisson={np.sum(np.array(model_params['model_selected']) == 'Poisson')}, " +
                  f"NB={np.sum(np.array(model_params['model_selected']) == 'NB')}, " +
                  f"ZIP={np.sum(np.array(model_params['model_selected']) == 'ZIP')}, " +
                  f"ZINB={np.sum(np.array(model_params['model_selected']) == 'ZINB')}")
            
            # Report boundary constraint effectiveness
            n_genes_clipped = np.sum(np.max(new_counts, axis=0) >= boundary_per_gene * 0.99)  # Close to boundary
            print(f"Boundary constraint applied to {n_genes_clipped}/{n_genes} genes")
            print(f"Final count range: [{np.min(new_counts):.1f}, {np.max(new_counts):.1f}]")
        
        return new_counts
    
    def _fallback_reference_based_simulation(self, reference_adata, boundary_multiplier=1.1):
        """Fallback method when proper parameters are not available.
        
        Args:
            boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max)
        """
        reference_matrix = reference_adata.X.toarray() if hasattr(reference_adata.X, 'toarray') else reference_adata.X.copy()
        
        # Calculate boundary using the tunable multiplier
        max_counts_per_gene = np.max(reference_matrix, axis=0)
        boundary_per_gene = max_counts_per_gene * boundary_multiplier
        
        # Add some biological variation while preserving overall structure
        noise_factor = 0.1
        biological_noise = np.random.gamma(2, 0.5, reference_matrix.shape)
        new_counts = reference_matrix * biological_noise * (1 + noise_factor * np.random.randn(*reference_matrix.shape))
        
        # Ensure non-negative and integer counts
        new_counts = np.maximum(new_counts, 0)
        new_counts = np.round(new_counts).astype(np.float32)
        
        # Apply boundary constraint: resample until all values within 110% boundary
        for gene_idx in range(new_counts.shape[1]):
            gene_boundary = boundary_per_gene[gene_idx]
            violations_mask = new_counts[:, gene_idx] > gene_boundary
            n_violations = np.sum(violations_mask)
            
            if n_violations > 0:
                violation_indices = np.where(violations_mask)[0]
                max_resample_attempts = 50  # Fewer attempts for fallback method
                
                for attempt in range(max_resample_attempts):
                    if n_violations == 0:
                        break
                    
                    # Use gamma distribution resampling for biological variation
                    shape = 2
                    scale = gene_boundary / (shape * 2)  # Scale to keep mean around boundary/2
                    resampled_values = np.random.gamma(shape, scale, size=n_violations)
                    
                    # Only keep values within boundary
                    valid_mask = resampled_values <= gene_boundary
                    valid_values = resampled_values[valid_mask]
                    n_valid = len(valid_values)
                    
                    if n_valid > 0:
                        update_indices = violation_indices[:n_valid]
                        new_counts[update_indices, gene_idx] = valid_values.astype(np.float32)
                        violation_indices = violation_indices[n_valid:]
                        n_violations = len(violation_indices)
                
                # Final fallback: uniform sampling within boundary
                if n_violations > 0:
                    uniform_values = np.random.uniform(0, gene_boundary, size=n_violations)
                    new_counts[violation_indices, gene_idx] = uniform_values.astype(np.float32)
        
        return new_counts
    
    def _apply_simple_parameter_assignment(self, reference_adata, model_params, verbose=False):
        """Fallback method for environments missing optional assignment dependencies."""
        if verbose:
            print("Using simplified parameter assignment (fallback mode)...")
        
        simulated_adata = reference_adata.copy()
        
        # Apply some basic variation to the reference data
        reference_matrix = simulated_adata.X.toarray() if hasattr(simulated_adata.X, 'toarray') else simulated_adata.X.copy()
        
        # Add controlled biological variation
        variation = np.random.gamma(1.2, 0.8, reference_matrix.shape)
        simulated_matrix = reference_matrix * variation
        simulated_matrix = np.maximum(simulated_matrix, 0)
        
        simulated_adata.X = simulated_matrix.astype(np.float32)
        simulated_adata.uns['simulation_method'] = 'simple_fallback'
        
        return simulated_adata
    
    def simulate_slice(self, **kwargs):
        """Convenience method for single slice simulation with parameter validation."""
        return self.simulate(**kwargs)

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
            "simulation_mode": kwargs.get("simulation_mode", "generative"),
            "quantile_calibration": kwargs.get("quantile_calibration"),
            "random_seed": kwargs.get("random_seed"),
        }
        self.fit_model(**fit_kwargs)
        simulated = self.simulate(
            num_simulation_cores=kwargs.get("num_simulation_cores", 12),
            verbose=kwargs.get("verbose", True),
            clip_overshoot_factor=kwargs.get("clip_overshoot_factor", 0.1),
            boundary_multiplier=kwargs.get("boundary_multiplier", 1.1),
            random_seed=kwargs.get("random_seed"),
        )
        simulated.uns["annotation_key"] = annotation_key
        return simulated
    

def simulate_single_slice(adata: ad.AnnData, visualize_fits: bool = False, num_simulation_cores: int = 12, verbose: bool = True, clip_overshoot_factor: float = 0.1, use_real_stats_directly: bool = False, annotation_key: str = None, use_heuristic_search: bool = False, min_accepted_error: float = 0.005, assignment_weights: dict = None, screening_pool_size: int = 1000, top_n_to_fully_evaluate: int = 10, n_jobs: int = -1, alteration_config=None, boundary_multiplier: float = 1.1, simulation_mode: str = 'generative', quantile_calibration=None, random_seed: int = None, hybrid_alpha: float = 0.2) -> ad.AnnData:
    """
    Run single-slice simulation with explicit generative or empirical semantics.
    
    Args:
        boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max).
            - 1.0: Strict boundary at reference maximum
            - 1.1: Allow 10% overshoot (default)
            - 1.5: Allow 50% overshoot for more variation
            - 2.0: Allow 100% overshoot for high variation
        alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions.
            Example:
                from FEAST.modeling.marginal_alteration import AlterationConfig
                config = AlterationConfig(
                    mean_fold_change=2.0,      # Double gene expression means
                    variance_fold_change=1.5,  # Increase variance by 50%
                    apply_to_mean=True,
                    apply_to_variance=True
                )
        simulation_mode: "generative" by default, or "empirical" for strict controlled alteration.
        quantile_calibration: "raw" for iid uniform quantiles, "reference_rank" for reference-ranked quantiles,
            or "auto" to use the mode default.
        random_seed: Optional seed for reproducible generative sampling and quantile decoding.
        Other parameters: See individual parameter documentation in fit_model() and simulate() methods.
    """
    simulation_mode = resolve_simulation_mode(simulation_mode)
    if verbose: print("Starting comprehensive single slice simulation...")
    adata = adata.copy()
    safe_calculate_qc_metrics(adata, verbose=verbose)
    simulator = SpatialSimulator(adata)
    
    # Combine heuristic search parameters into kwargs to pass them down easily
    heuristic_kwargs = {
        'use_heuristic_search': use_heuristic_search,
        'min_accepted_error': min_accepted_error,
        'assignment_weights': assignment_weights,
        'screening_pool_size': screening_pool_size,
        'top_n_to_fully_evaluate': top_n_to_fully_evaluate,
        'n_jobs': n_jobs,
        'simulation_mode': simulation_mode,
        'quantile_calibration': quantile_calibration,
        'random_seed': random_seed,
        'hybrid_alpha': hybrid_alpha,
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
        )
        
    if verbose: print(f"\nSimulation completed successfully!")
    return simulated_adata
