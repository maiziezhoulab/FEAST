import numpy as np
import pandas as pd
import scanpy as sc
import alphashape
from shapely.geometry import Point, MultiPolygon
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

global device   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_low_resolution_grid(spatial_coords, downsampling_factor=0.25, grid_type='hexagonal'):
    """Create a lower resolution grid based on original spatial coordinates."""
    x_min, y_min = np.min(spatial_coords, axis=0)
    x_max, y_max = np.max(spatial_coords, axis=0)
    
    base_spots = len(spatial_coords)
    target_spots = int(base_spots * downsampling_factor)
    
    if grid_type == 'square':
        # Create a square grid with fewer points
        grid_size = int(np.sqrt(target_spots))
        x_points = np.linspace(x_min, x_max, grid_size)
        y_points = np.linspace(y_min, y_max, grid_size)
        x_centers = (x_points[:-1] + x_points[1:]) / 2
        y_centers = (y_points[:-1] + y_points[1:]) / 2
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)
        grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    elif grid_type == 'hexagonal':
        # Create a hexagonal grid with fewer points
        spacing = np.sqrt((x_max - x_min) * (y_max - y_min) / target_spots)
        hex_centers = []
        for i in range(int((y_max - y_min) / spacing) + 1):
            for j in range(int((x_max - x_min) / spacing) + 1):
                x = x_min + j * spacing + (i % 2) * spacing / 2
                y = y_min + i * spacing * np.sqrt(3) / 2
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    hex_centers.append([x, y])
        grid_coords = np.array(hex_centers)
    
    elif grid_type == 'kmeans':
        # Use KMeans to create spot clusters
        kmeans = KMeans(n_clusters=target_spots, random_state=42, n_init=10)
        kmeans.fit(spatial_coords)
        grid_coords = kmeans.cluster_centers_
    
    else:
        raise ValueError(f"Unsupported grid type: {grid_type}")
    
    return grid_coords

def filter_grid_to_tissue_shape(original_coords, grid_coords, alpha=0.01):
    """Filter grid points to only include those within the tissue shape."""
    alpha_shape = alphashape.alphashape(original_coords, alpha)
    if isinstance(alpha_shape, MultiPolygon):
        polygons = list(alpha_shape.geoms)
    else:
        polygons = [alpha_shape]
    
    mask = np.array([
        any(polygon.contains(Point(p)) for polygon in polygons) for p in grid_coords
    ])
    
    return grid_coords[mask]

def assign_original_spots_to_grid(original_coords, grid_coords):
    """Assign each original spot to the nearest low-resolution grid point."""
    tree = cKDTree(grid_coords)
    distances, assignments = tree.query(original_coords)
    return assignments

def calculate_cell_type_proportions_for_lowres(cell_types, assignments, n_lowres_spots):
    """Calculate cell type proportions for each low-resolution spot."""
    unique_cell_types = np.unique(cell_types)
    proportions = np.zeros((n_lowres_spots, len(unique_cell_types)))
    
    # For each low-resolution spot, calculate cell type proportions
    for i in range(n_lowres_spots):
        # Find which original spots are assigned to this low-resolution spot
        mask = assignments == i
        if np.sum(mask) > 0:
            # Get cell types of assigned spots
            spot_cell_types = cell_types[mask]
            # Calculate proportions
            for j, cell_type in enumerate(unique_cell_types):
                proportions[i, j] = np.sum(spot_cell_types == cell_type) / len(spot_cell_types)
    
    # Create DataFrame with cell type proportions
    proportions_df = pd.DataFrame(proportions, columns=unique_cell_types)
    
    return proportions_df

def aggregate_gene_expression(expression_matrix, assignments, n_lowres_spots):
    """
    Aggregate gene expression from original spots to low-resolution spots.
    
    This function sums the gene expression from all original spots assigned to
    each low-resolution spot.
    """
    # Convert to PyTorch tensors
    expression_tensor = torch.tensor(expression_matrix, dtype=torch.float32, device=device)
    assignments_tensor = torch.tensor(assignments, dtype=torch.long, device=device)
    
    # Initialize output tensor
    n_genes = expression_matrix.shape[1]
    aggregated_counts = torch.zeros((n_lowres_spots, n_genes), device=device)
    
    # For each low-resolution spot, sum expression from assigned original spots
    for i in range(n_lowres_spots):
        mask = assignments_tensor == i
        if torch.sum(mask) > 0:
            aggregated_counts[i] = torch.sum(expression_tensor[mask], dim=0)
    
    return aggregated_counts.cpu().numpy()

def create_deconvolution_benchmark_data(adata, downsampling_factor=0.25, grid_type='hexagonal', cell_type_key=None, alpha=0.01):
    """
    Create lower-resolution ST data for deconvolution benchmarking with ground truth proportions.
    
    Parameters:
    -----------
    adata : AnnData
        Original ST data
    downsampling_factor : float
        Factor to reduce resolution (0.25 = 4x fewer spots)
    grid_type : str
        Grid type ('hexagonal', 'square', or 'kmeans')
    cell_type_key : str
        Key in adata.obs containing cell type labels
    alpha : float
        Alpha value for the alphashape algorithm
        
    Returns:
    --------
    AnnData
        Lower-resolution ST data with ground truth proportions
    """
    # Extract spatial coordinates and expression matrix
    original_coords = adata.obsm['spatial']
    
    # Convert expression matrix to dense if it's sparse
    if isinstance(adata.X, np.ndarray):
        expression_matrix = adata.X
    else:
        expression_matrix = adata.X.toarray()
    
    # Generate low-resolution grid
    lowres_grid = create_low_resolution_grid(original_coords, downsampling_factor, grid_type)
    
    # Filter grid to tissue shape
    lowres_grid = filter_grid_to_tissue_shape(original_coords, lowres_grid, alpha)
    
    # Assign original spots to low-resolution grid points
    assignments = assign_original_spots_to_grid(original_coords, lowres_grid)
    
    # Aggregate gene expression
    aggregated_counts = aggregate_gene_expression(expression_matrix, assignments, len(lowres_grid))
    
    # Create new AnnData object
    lowres_adata = sc.AnnData(X=aggregated_counts, var=adata.var.copy())
    lowres_adata.obsm['spatial'] = lowres_grid
    
    # Calculate and store ground truth cell type proportions if cell type information is available
    if cell_type_key is not None and cell_type_key in adata.obs:
        cell_types = adata.obs[cell_type_key].values
        proportions_df = calculate_cell_type_proportions_for_lowres(
            cell_types, 
            assignments, 
            len(lowres_grid)
        )

        # Store proportions directly in the AnnData object
        lowres_adata.obsm['cell_type_proportions'] = proportions_df.values
        lowres_adata.uns['cell_type_names'] = proportions_df.columns.values
    
    # Store assignment information
    lowres_adata.uns['spot_assignments'] = assignments
    
    # Add metadata
    lowres_adata.uns['benchmark_params'] = {
        'original_spots': len(original_coords),
        'low_res_spots': len(lowres_grid),
        'downsampling_factor': downsampling_factor,
        'grid_type': grid_type,
        'has_ground_truth': cell_type_key is not None and cell_type_key in adata.obs
    }
    
    return lowres_adata