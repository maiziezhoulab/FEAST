import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

from .parameter_cloud import GeneParameterSimulator, convert_params_for_new_simulator

def safe_calculate_qc_metrics(adata, verbose=False):
    try:
        if adata.n_vars > 0 and adata.n_obs > 0: sc.pp.calculate_qc_metrics(adata, percent_top=[20, 50, 100] if adata.n_vars > 100 else [50], inplace=True, log1p=False)
    except Exception as e:
        if verbose: print(f"Warning: QC calculation failed ({e}), using basic metrics only")
        adata.obs['total_counts'] = np.asarray(adata.X.sum(axis=1)).flatten()
        adata.obs['n_genes_by_counts'] = np.asarray((adata.X > 0).sum(axis=1)).flatten()
        adata.var['total_counts'] = np.asarray(adata.X.sum(axis=0)).flatten()
        adata.var['n_cells_by_counts'] = np.asarray((adata.X > 0).sum(axis=0)).flatten()

def run_parameter_cloud_fitting(adata, visualize_fits=False, use_heuristic_search=True, min_accepted_error=0.5, assignment_weights=None, screening_pool_size=100, top_n_to_fully_evaluate=10, n_jobs=-1, alteration_config=None):
    """
    UPDATED: Calls the new two-stage, parallelized heuristic search with optional marginal distribution alteration.
    
    Args:
        alteration_config (AlterationConfig or dict, optional): Configuration for altering marginal distributions
    """
    print("\n>>> Entering STANDARD fitting pipeline: parameter_cloud <<<")
    
    if assignment_weights is None:
        assignment_weights = {'mean': 1, 'variance': 1, 'zero_prop': 1.0}

    simulator = GeneParameterSimulator()
    simulator.fit(adata, visualize_fits=visualize_fits)
    
    # Apply marginal distribution alterations if requested
    if alteration_config is not None:
        simulator.alter_marginal_distributions(alteration_config=alteration_config, verbose=True)
    
    if use_heuristic_search:
        print("--- Activating Boosted Heuristic Optimization Mode ---")
        assigned_synthetic_params = simulator.run_heuristic_search(
            n_genes=adata.n_vars,
            min_accepted_error=min_accepted_error,
            assignment_weights=assignment_weights,
            screening_pool_size=screening_pool_size,
            top_n_to_fully_evaluate=top_n_to_fully_evaluate,
            n_jobs=n_jobs
        )
    else:
        print("--- Using Standard Single-Shot Assignment ---")
        synthetic_params = simulator.simulate(n_genes=adata.n_vars)
        assigned_synthetic_params = simulator.assign_to_genes(
            synthetic_params, 
            weights=assignment_weights
        )

    model_params = convert_params_for_new_simulator(assigned_synthetic_params)
    model_params['simulation_evaluation'] = {'source': 'parameter_cloud_v2_robust_boosted'}
    print(">>> Exiting parameter_cloud pipeline <<<\n")
    return model_params

def run_direct_fitting_from_real_stats(adata):
    """This diagnostic pipeline remains unchanged."""
    print("\n>>> Entering DIAGNOSTIC fitting pipeline: Using REAL stats directly <<<")
    simulator = GeneParameterSimulator()
    simulator.fit_statistics_only(adata)
    real_stats_for_conversion = simulator.original_stats.reset_index().rename(columns={'index': 'gene_id'})
    model_params = convert_params_for_new_simulator(real_stats_for_conversion)
    model_params['simulation_evaluation'] = {'source': 'direct_from_real_stats'}
    print(">>> Diagnostic fitting complete <<<\n")
    return model_params


class SpatialSimulator:
    def __init__(self, reference_adata: ad.AnnData, model_params: dict = None):
        if 'spatial' not in reference_adata.obsm: raise ValueError("Reference AnnData must contain 'spatial' coordinates.")
        self.reference_adata = reference_adata.copy() 
        self.reference_adata.var_names_make_unique()
        self.reference_adata.obs_names_make_unique()
        self._model_params = model_params

    def fit_model(self, visualize_fits: bool = False, use_real_stats_directly: bool = False, use_heuristic_search: bool = False, min_accepted_error: float = 0.5, assignment_weights: dict = None, screening_pool_size: int = 100, top_n_to_fully_evaluate: int = 10, n_jobs: int = -1, alteration_config=None) -> 'SpatialSimulator':
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
                alteration_config=alteration_config
            )
        return self
    
    def set_model_params(self, model_params: dict):
        """Set model parameters directly (useful for parameter cloud interpolation)."""
        self._model_params = model_params
        return self
    
    def get_model_params(self):
        """Get current model parameters."""
        return self._model_params
    
    def simulate(self, sigma: float = 1.0, follower_sigma_factor: float = 0.2, num_simulation_cores: int = 12, verbose: bool = True, clip_overshoot_factor: float = 0.0, boundary_multiplier: float = 1.1) -> ad.AnnData:
        """
        Args:
            sigma (float): The key gentleness parameter. Controls the magnitude of primary spatial noise.
                - sigma=0: Perfect pattern preservation (zero spatial change).
                - sigma=0.5-1.5: "Gentle" mode, introduces subtle, local variations.
                - sigma>2.0: "Exploratory" mode, introduces more significant changes.
            follower_sigma_factor (float): Now used as correlation threshold for debugging.
                The algorithm automatically uses correlation-guided interpolation for followers.
            num_simulation_cores (int): Number of cores for simulation (legacy parameter).
            verbose (bool): If True, prints progress updates.
            clip_overshoot_factor (float): Factor to clip max expression values relative to reference.
            boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max).
        """
        if self._model_params is None:
            raise ValueError("Model parameters not set. Call fit_model() first or provide model_params in constructor.")
        
        if verbose:
            if sigma == 0:
                print("Generating simulated data using Perfect Pattern Preservation (sigma=0)...")
            elif sigma <= 1.5:
                print(f"Generating simulated data using Gentle G-SRBA with Correlation-Guided Followers (sigma={sigma})...")
            else:
                print(f"Generating simulated data using Exploratory G-SRBA with Correlation-Guided Followers (sigma={sigma})...")
        
        # Apply the sophisticated G-SRBA algorithm with correlation-guided follower assignment
        simulated_adata = self._apply_guided_stochastic_assignment(
            reference_adata=self.reference_adata,
            model_params=self._model_params,
            sigma=sigma,
            follower_sigma_factor=follower_sigma_factor,
            verbose=verbose,
            clip_overshoot_factor=clip_overshoot_factor,
            boundary_multiplier=boundary_multiplier
        )
        
        if verbose:
            print(f"Correlation-Guided G-SRBA simulation complete for {simulated_adata.n_obs} spots and {simulated_adata.n_vars} genes")
        
        safe_calculate_qc_metrics(simulated_adata)
        return simulated_adata
    
    def _apply_guided_stochastic_assignment(self, reference_adata, model_params, sigma=1.0, follower_sigma_factor=0.2, verbose=True, clip_overshoot_factor=0.0, boundary_multiplier=1.1, n_modules=30, n_neighbors=6):
        """
        REFINED: Implements the Guided Stochastic Rank-Based Assignment (G-SRBA) algorithm with tunable gentleness.
        
        Args:
            boundary_multiplier (float): Multiplier for maximum count boundary constraint (default 1.1 = 110% of reference max)
        
        The "gentleness" is controlled by sigma:
        - sigma=0: Zero spatial change, perfect pattern preservation
        - sigma=0.5-1.5: Gentle mode with subtle local variations  
        - sigma>2.0: Exploratory mode with significant spatial changes
        """
        try:
            from sklearn.decomposition import NMF
            from sklearn.neighbors import kneighbors_graph
            from scipy.sparse import csgraph
        except ImportError as e:
            if verbose:
                print(f"Warning: Required libraries not available ({e}). Using simplified assignment.")
            return self._apply_simple_parameter_assignment(reference_adata, model_params, verbose)
        
        if verbose:
            print("Applying Guided Stochastic Rank-Based Assignment (G-SRBA)...")
        
        # Extract data matrices
        reference_matrix = reference_adata.X.toarray() if hasattr(reference_adata.X, 'toarray') else reference_adata.X.copy()
        spatial_coords = reference_adata.obsm['spatial']
        
        # SPECIAL CASE: Perfect Pattern Preservation (sigma=0)
        if sigma == 0:
            if verbose:
                print("Applying perfect pattern preservation - maintaining exact spatial structure...")
            
            # Generate new counts from parameters but preserve exact spatial ranking
            new_counts = self._generate_counts_from_parameters(reference_adata, model_params, verbose)
            
            # For perfect preservation, assign new counts in exact same spatial order
            # This applies statistical parameter changes while preserving spatial structure perfectly
            simulated_matrix = np.zeros_like(reference_matrix, dtype=np.float32)
            
            for gene_idx in range(reference_matrix.shape[1]):
                # Get the spatial ranking from original data
                original_spatial_ranks = np.argsort(reference_matrix[:, gene_idx])
                # Get the new expression values sorted by magnitude
                new_values_sorted = np.sort(new_counts[:, gene_idx])
                # Assign new values in same spatial order as original
                simulated_matrix[original_spatial_ranks, gene_idx] = new_values_sorted
            
            simulated_adata = ad.AnnData(
                X=simulated_matrix.astype(np.float32),
                obs=reference_adata.obs.copy(),
                var=reference_adata.var.copy(),
                obsm={'spatial': spatial_coords.copy()}
            )
            
            simulated_adata.uns['simulation_method'] = 'Perfect_Pattern_Preservation'
            simulated_adata.uns['simulation_params'] = {
                'sigma': sigma,
                'follower_sigma_factor': follower_sigma_factor,
                'clip_overshoot_factor': clip_overshoot_factor,
                'preservation_mode': 'parameter_aware_perfect_ranking'
            }
            
            return simulated_adata
        
        # Generate new counts based on model parameters
        new_counts = self._generate_counts_from_parameters(reference_adata, model_params, verbose, boundary_multiplier)
        
        n_spots, n_genes = reference_matrix.shape
        
        # Phase 1: Discover Co-expression Structure
        if verbose:
            print(f"Phase 1: Finding {n_modules} gene co-expression modules...")
        
        gene_modules, leader_genes_indices = self._find_gene_modules(new_counts, n_modules, verbose)
        
        # Phase 2: Prepare Spatial Structure
        if verbose:
            print(f"Phase 2: Building spatial graph with {n_neighbors} neighbors...")
        
        spatial_smoother = self._create_spatial_smoother(spatial_coords, n_neighbors)
        
        # Phase 3: G-SRBA Core Algorithm
        if verbose:
            if sigma == 0:
                print("Phase 3: Applying perfect pattern preservation...")
            elif sigma <= 1.5:
                print(f"Phase 3: Applying gentle stochastic assignment (sigma={sigma})...")
            else:
                print(f"Phase 3: Applying exploratory stochastic assignment (sigma={sigma})...")
        
        S_final = self._guided_assignment_core(
            reference_matrix, new_counts, spatial_smoother, 
            gene_modules, leader_genes_indices, sigma, follower_sigma_factor, verbose
        )
        
        # Apply clipping if requested
        if clip_overshoot_factor > 0:
            max_ref_counts = np.max(reference_matrix, axis=0)
            clip_max = max_ref_counts * (1 + clip_overshoot_factor)
            S_final = np.clip(S_final, 0, clip_max)
        
        # Create output AnnData
        simulated_adata = ad.AnnData(
            X=S_final.astype(np.float32),
            obs=reference_adata.obs.copy(),
            var=reference_adata.var.copy(),
            obsm={'spatial': spatial_coords.copy()}
        )
        
        # Add metadata about the simulation
        if sigma == 0:
            method_name = 'Perfect_Pattern_Preservation'
        elif sigma <= 1.5:
            method_name = 'Gentle_Correlation_Guided_G-SRBA'
        else:
            method_name = 'Exploratory_Correlation_Guided_G-SRBA'
            
        simulated_adata.uns['simulation_method'] = method_name
        simulated_adata.uns['simulation_params'] = {
            'n_modules': n_modules,
            'n_neighbors': n_neighbors,
            'sigma': sigma,
            'follower_method': 'correlation_guided_interpolation',
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
                               gene_modules, leader_genes_indices, sigma, follower_sigma_factor, verbose=False):
        """
        REFINED Core G-SRBA Algorithm with Correlation-Guided Interpolation for Followers.
        
        Args:
            sigma (float): Controls the magnitude of spatial noise for leaders
            follower_sigma_factor (float): Legacy parameter, now used as correlation threshold
        """
        n_spots, n_genes = reference_matrix.shape
        S_final = np.zeros_like(reference_matrix, dtype=np.float32)
        
        # Precompute rank orderings
        reference_ranks = np.argsort(reference_matrix, axis=0)
        new_counts_ranks = np.argsort(new_counts, axis=0)
        
        # Calculate gene-gene correlation matrix for guidance
        if verbose:
            print("Calculating gene-gene correlation matrix for correlation-guided interpolation...")
        
        # Use pandas for robust correlation calculation
        gene_corr = pd.DataFrame(new_counts).corr().values
        
        # Track assigned genes to handle overlaps
        assigned_genes = set()
        
        for i, module in enumerate(gene_modules):
            if not module or i >= len(leader_genes_indices):
                continue
            
            leader_gene_idx = leader_genes_indices[i]
            
            if leader_gene_idx >= n_genes or leader_gene_idx in assigned_genes:
                continue
            
            # Step A: Assign leader gene with controlled spatial noise
            base_noise = np.random.randn(n_spots)
            smooth_noise = spatial_smoother.dot(base_noise)
            
            # Normalize the smooth noise to have controlled magnitude
            if np.std(smooth_noise) > 0:
                smooth_noise = smooth_noise / np.std(smooth_noise)
            
            perfect_map = np.arange(n_spots, dtype=np.float64)
            
            # Apply sigma-controlled perturbation with ULTRA-CONSERVATIVE scaling
            if sigma > 0:
                # ULTRA-CONSERVATIVE: For small sigma, use extremely gentle scaling
                # Target: sigma=0.05 should preserve ~80% spatial correlation
                
                if sigma <= 0.05:
                    # For very tiny sigma: use minimal perturbation
                    # sigma=0.05 -> noise_scale = 0.05 * 10 * 0.1 = 0.05 (extremely gentle)
                    noise_scale = sigma * np.sqrt(n_spots) * 0.1
                elif sigma <= 0.1:
                    # For small sigma: slightly more perturbation
                    # sigma=0.1 -> noise_scale = 0.1 * 10 * 0.2 = 0.2
                    noise_scale = sigma * np.sqrt(n_spots) * 0.2
                else:
                    # For larger sigma: more traditional scaling
                    noise_scale = sigma * n_spots * 0.3
                
                perturbed_map = perfect_map + smooth_noise * noise_scale
                shuffled_indices = np.argsort(perturbed_map)
            else:
                shuffled_indices = np.arange(n_spots, dtype=int)
            
            # Assign leader gene
            assigned_indices = new_counts_ranks[shuffled_indices, leader_gene_idx]
            S_final[reference_ranks[:, leader_gene_idx], leader_gene_idx] = \
                new_counts[assigned_indices, leader_gene_idx]
            
            assigned_genes.add(leader_gene_idx)
            
            # Step B: Assign follower genes with Correlation-Guided Interpolation
            leader_pattern_ranks = np.argsort(S_final[:, leader_gene_idx])
            
            for follower_gene_idx in module:
                if follower_gene_idx == leader_gene_idx or follower_gene_idx >= n_genes:
                    continue
                if follower_gene_idx in assigned_genes:
                    continue
                
                # 1. Get the correlation weight (how much should this gene follow the leader?)
                correlation = gene_corr[leader_gene_idx, follower_gene_idx]
                # Use absolute correlation and clip to ensure it's a valid weight (0 to 1)
                correlation_weight = np.clip(abs(correlation), 0, 1)
                
                # 2. SIMPLE INTERPOLATION: Blend the two rank patterns
                # Pattern A: The leader's new spatial pattern
                pattern_A_ranks = leader_pattern_ranks.astype(np.float64)
                # Pattern B: The follower's original spatial pattern from reference
                pattern_B_ranks = reference_ranks[:, follower_gene_idx].astype(np.float64)
                
                # 3. Linear interpolation based on correlation strength
                # High correlation = more following of leader, low correlation = more independent
                interpolated_rank_map = (pattern_A_ranks * correlation_weight) + \
                                      (pattern_B_ranks * (1 - correlation_weight))
                
                # 4. Convert to assignment indices
                follower_shuffled_indices = np.argsort(interpolated_rank_map)
                
                # 5. Assign the follower's counts
                assigned_indices_follower = new_counts_ranks[follower_shuffled_indices, follower_gene_idx]
                S_final[:, follower_gene_idx] = \
                    new_counts[assigned_indices_follower, follower_gene_idx]
                
                assigned_genes.add(follower_gene_idx)
        
        # Handle any unassigned genes with simple SRBA
        unassigned_genes = set(range(n_genes)) - assigned_genes
        if unassigned_genes and verbose:
            print(f"Assigning {len(unassigned_genes)} remaining genes with gentle SRBA...")
        
        for gene_idx in unassigned_genes:
            base_noise = np.random.randn(n_spots)
            smooth_noise = spatial_smoother.dot(base_noise)
            
            # Normalize noise
            if np.std(smooth_noise) > 0:
                smooth_noise = smooth_noise / np.std(smooth_noise)
            
            perfect_map = np.arange(n_spots, dtype=np.float64)
            
            # Apply reduced noise for unassigned genes
            if sigma > 0:
                # Use ultra-conservative scaling for unassigned genes too
                if sigma <= 0.05:
                    reduced_noise_scale = 0.2 * sigma * np.sqrt(n_spots) * 0.1
                elif sigma <= 0.1:
                    reduced_noise_scale = 0.2 * sigma * np.sqrt(n_spots) * 0.2
                else:
                    reduced_noise_scale = 0.2 * sigma * n_spots * 0.3
                
                perturbed_map = perfect_map + smooth_noise * reduced_noise_scale
                shuffled_indices = np.argsort(perturbed_map)
            else:
                shuffled_indices = np.arange(n_spots, dtype=int)
            
            assigned_indices = new_counts_ranks[shuffled_indices, gene_idx]
            S_final[reference_ranks[:, gene_idx], gene_idx] = \
                new_counts[assigned_indices, gene_idx]
        
        return S_final
    
    def _apply_local_neighbor_swapping(self, perfect_map, spatial_smoother, sigma, n_spots):
        """
        FUNDAMENTAL APPROACH: For very small sigma, instead of adding noise,
        perform controlled local swaps between spatial neighbors.
        This preserves global spatial structure while introducing minimal local variation.
        """
        perturbed_map = perfect_map.copy()
        
        # Calculate number of swaps based on sigma
        # For sigma=0.05, we want ~5% of spots to participate in local swaps
        max_swaps = int(sigma * n_spots * 10)  # sigma=0.05 -> ~5 swaps for 100 spots
        
        if max_swaps == 0:
            return perturbed_map
        
        # Get spatial adjacency information
        # Convert spatial_smoother to adjacency list for efficiency
        adjacency = spatial_smoother.tocoo()
        
        # Find all spatial neighbor pairs
        neighbor_pairs = []
        for i, j in zip(adjacency.row, adjacency.col):
            if i < j:  # Avoid duplicates
                neighbor_pairs.append((i, j))
        
        if len(neighbor_pairs) == 0:
            return perturbed_map
        
        # Randomly select pairs to swap
        n_swaps = min(max_swaps, len(neighbor_pairs))
        swap_pairs = np.random.choice(len(neighbor_pairs), size=n_swaps, replace=False)
        
        # Perform the swaps
        for swap_idx in swap_pairs:
            i, j = neighbor_pairs[swap_idx]
            # Swap the ranking positions of these two neighbors
            perturbed_map[i], perturbed_map[j] = perturbed_map[j], perturbed_map[i]
        
        return perturbed_map
    
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
        """Fallback method if G-SRBA dependencies are not available."""
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
    

def simulate_single_slice(adata: ad.AnnData, sigma: float = 0, follower_sigma_factor: float = 0, visualize_fits: bool = False, num_simulation_cores: int = 12, verbose: bool = True, clip_overshoot_factor: float = 0.1, use_real_stats_directly: bool = False, annotation_key: str = None, use_heuristic_search: bool = False, min_accepted_error: float = 0.005, assignment_weights: dict = None, screening_pool_size: int = 1000, top_n_to_fully_evaluate: int = 10, n_jobs: int = -1, alteration_config=None, boundary_multiplier: float = 1.1, **kwargs) -> ad.AnnData:
    """
    UPDATED: G-SRBA with Correlation-Guided Interpolation for follower genes and marginal distribution alteration.
    
    Args:
        sigma (float): The key gentleness parameter. Controls the magnitude of primary spatial noise.
            - sigma=0: Perfect pattern preservation (zero spatial change).
            - sigma=0.5-1.5: "Gentle" mode, introduces subtle, local variations.
            - sigma>2.0: "Exploratory" mode, introduces more significant changes.
        follower_sigma_factor (float): Legacy parameter, now automatically uses correlation-guided 
            interpolation for followers based on gene-gene correlations in the target distribution.
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
        Other parameters: See individual parameter documentation in fit_model() and simulate() methods.
    """
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
        'n_jobs': n_jobs
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
            **heuristic_kwargs, # Pass all heuristic controls
            **kwargs
        )


    else:
        if use_real_stats_directly:
            if verbose: print("--- RUNNING IN DIAGNOSTIC MODE (USING REAL STATS) ---")
        elif use_heuristic_search:
            if verbose: print("--- RUNNING IN BOOSTED HEURISTIC OPTIMIZATION MODE ---")
        else:
            if verbose: 
                if sigma == 0:
                    print("--- RUNNING IN STANDARD MODE WITH PERFECT PATTERN PRESERVATION ---")
                elif sigma <= 1.5:
                    print(f"--- RUNNING IN STANDARD MODE WITH GENTLE G-SRBA (sigma={sigma}) ---")
                else:
                    print(f"--- RUNNING IN STANDARD MODE WITH EXPLORATORY G-SRBA (sigma={sigma}) ---")
        
        simulator.fit_model(
            visualize_fits=visualize_fits, 
            use_real_stats_directly=use_real_stats_directly,
            alteration_config=alteration_config,  # Pass alteration configuration
            **heuristic_kwargs # Pass all heuristic controls
        )
        simulated_adata = simulator.simulate(
            sigma=sigma,
            follower_sigma_factor=follower_sigma_factor,
            num_simulation_cores=num_simulation_cores, 
            verbose=verbose, 
            clip_overshoot_factor=clip_overshoot_factor,
            boundary_multiplier=boundary_multiplier
        )
        
    if verbose: print(f"\nSimulation completed successfully!")
    return simulated_adata