import numpy as np
import scipy.spatial
import anndata as ad
from tps import ThinPlateSpline

class SpatialTransformer:
    """Base class for spatial transcriptomics data transformations with ground truth tracking."""
    
    def __init__(self, adata):
        if 'spatial' not in adata.obsm:
            raise ValueError("No spatial coordinates found in 'adata.obsm['spatial']'.")
        
        self.adata = adata.copy()
        self.spatial_data = self.adata.obsm['spatial'].copy()
        self.adata.obsm['spatial_original'] = self.spatial_data.copy()
    
    def transform(self):
        """Apply transformation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement transform()")
    
    def _minimal_internal_space(self, points):
        """Calculate minimum distance between any two points."""
        min_distance = float('inf')
        kdtree = scipy.spatial.KDTree(points)
        distances, _ = kdtree.query(points, k=2)
        return np.min(distances[:, 1])  # Second nearest neighbor (first is self)
    
    def cut(self, x_min, x_max, y_min, y_max):
        """Cut a rectangular section from the spatial data."""
        mask = (
            (self.spatial_data[:, 0] >= x_min) & (self.spatial_data[:, 0] <= x_max) &
            (self.spatial_data[:, 1] >= y_min) & (self.spatial_data[:, 1] <= y_max)
        )
        adata_cut = self.adata[mask].copy()
        return adata_cut
    
    def move(self, dx, dy):
        # Get original boundaries
        x_min, x_max = np.min(self.adata.obsm['spatial'][:, 0]), np.max(self.adata.obsm['spatial'][:, 0])
        y_min, y_max = np.min(self.adata.obsm['spatial'][:, 1]), np.max(self.adata.obsm['spatial'][:, 1])
        
        # Create a copy and move coordinates
        moved_adata = self.adata.copy()
        moved_coords = self.spatial_data + np.array([dx, dy])
        
        # Create mask for spots that remain within original boundaries
        in_bounds = (
            (moved_coords[:, 0] >= x_min) & (moved_coords[:, 0] <= x_max) &
            (moved_coords[:, 1] >= y_min) & (moved_coords[:, 1] <= y_max)
        )
        
        # Filter the AnnData object and update spatial coordinates
        moved_adata = moved_adata[in_bounds].copy()
        moved_adata.obsm['spatial'] = moved_coords[in_bounds]
        
        return moved_adata


class RotationTransformer:
    """Transform spatial data by rotation for both sequencing-based and imaging-based methods."""
    
    def __init__(self, adata):

        self.adata = adata
        self.spatial_data = adata.obsm['spatial'].copy()
    
    def _rotate(self, coords, angle_degrees, center_correction=0):
        theta = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        coords_centered = coords + center_correction
        rotated_coords = np.array([rotation_matrix.dot(point) for point in coords_centered])
        rotated_coords -= center_correction

        return rotated_coords
    
    def _generate_offset_grid(self, rows, cols, spacing, offset, x_range, y_range):
        """
        Generate a hexagonal grid of points.
        
        Parameters
        ----------
        rows : int
            Number of rows in the grid
        cols : int
            Number of columns in the grid
        spacing : float
            Horizontal spacing between adjacent points
        offset : float
            Vertical spacing between rows
        x_range : list
            [min_x, max_x] range for the grid
        y_range : list
            [min_y, max_y] range for the grid
            
        Returns
        -------
        numpy.ndarray
            Array of grid coordinates
        """
        grid = []
        for row in range(rows):
            for col in range(cols):
                x = x_range[0] + col * spacing * 2 + (row % 2) * spacing
                y = y_range[0] + row * offset
                grid.append([x, y])
        return np.array(grid)
    
    def _find_nearest_grid_points(self, rotated_spots, grid):
        """
        Find nearest grid points for each rotated spot using a KD-tree.
        
        Parameters
        ----------
        rotated_spots : numpy.ndarray
            Array of rotated spot coordinates
        grid : numpy.ndarray
            Array of grid coordinates
            
        Returns
        -------
        numpy.ndarray
            Indices of nearest grid points
        """
        import scipy.spatial
        
        kdtree = scipy.spatial.KDTree(grid)
        _, nearest_indices = kdtree.query(rotated_spots, k=1)
        
        return nearest_indices
    
    def _get_spatial_bounds(self, coords):
        """
        Get the min/max bounds of the spatial coordinates.
        
        Parameters
        ----------
        coords : numpy.ndarray
            Array of coordinates
            
        Returns
        -------
        tuple
            (min_x, max_x, min_y, max_y)
        """
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        return np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)
    
    def _filter_points_in_range(self, coords, original_bounds):
        """
        Filter points to keep only those within the given bounds.
        
        Parameters
        ----------
        coords : numpy.ndarray
            Array of coordinates
        original_bounds : tuple
            (min_x, max_x, min_y, max_y) bounds
            
        Returns
        -------
        numpy.ndarray
            Boolean mask of points within bounds
        """
        min_x, max_x, min_y, max_y = original_bounds
        
        in_bounds = (
            (coords[:, 0] >= min_x) & 
            (coords[:, 0] <= max_x) & 
            (coords[:, 1] >= min_y) & 
            (coords[:, 1] <= max_y)
        )
        
        return in_bounds
    
    def transform_sequencing(self, rotation_angle=0, center_correction=0, 
                             min_space=None, max_grid_size=10000):
        """
        Transform sequencing-based spatial data with rotation and grid alignment.
        
        Parameters
        ----------
        rotation_angle : float
            Rotation angle in degrees
        center_correction : float or array-like
            Center point for rotation
        min_space : float, optional
            Minimal spacing between spots. If None, will be calculated from data.

        Returns
        -------
        AnnData
            Transformed AnnData object with grid-aligned coordinates
        """
        print(f"Transforming sequencing-based data with {self.spatial_data.shape[0]} spots")
        
        # Get spatial bounds
        x_coords = self.spatial_data[:, 0]
        y_coords = self.spatial_data[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        x_range = [min_x, max_x]
        y_range = [min_y, max_y]
        
        if min_space is None:
            from scipy.spatial import distance_matrix
            
            # Calculate pairwise distances for a subset of points
            sample_size = min(1000, len(self.spatial_data))
            indices = np.random.choice(len(self.spatial_data), size=sample_size, replace=False)
            sampled_coords = self.spatial_data[indices]
            
            dist_matrix = distance_matrix(sampled_coords, sampled_coords)
            np.fill_diagonal(dist_matrix, np.inf)
            min_space = np.min(dist_matrix)
            print(f"Calculated minimal spacing: {min_space}")
        
        # Calculate grid parameters
        spacing = min_space / 2
        offset = min_space * np.sqrt(3) / 2
        
        # Estimate grid size
        rows_full = round((y_range[1] - y_range[0]) / offset) + 1
        cols_full = round((x_range[1] - x_range[0]) / (spacing * 2)) + 1
        estimated_grid_size = rows_full * cols_full
        

        rows, cols = rows_full, cols_full
        
        print(f"Generating grid with {rows}x{cols}={rows*cols} points")
        grid = self._generate_offset_grid(rows, cols, spacing, offset, x_range, y_range)
        
        # Rotate the coordinates
        print(f"Rotating spots by {rotation_angle} degrees")
        rotated_spots = self._rotate(self.spatial_data, rotation_angle, center_correction)
        
        print(f"Finding nearest grid points")
        nearest_indices = self._find_nearest_grid_points(rotated_spots, grid)
        new_spots = grid[nearest_indices]
        
        # Remove duplicates
        print("Removing duplicates")
        seen = {}
        mapping = []
        for i in range(len(new_spots)):
            spot_tuple = tuple(new_spots[i])
            if spot_tuple in seen:
                continue
            seen[spot_tuple] = 1
            mapping.append(i)
        
        print(f"Creating new AnnData with {len(mapping)} spots")
        new_adata = self.adata[mapping, :].copy()
        new_adata.obsm['spatial'] = new_spots[mapping]
        new_adata.obsm['spatial_rotated'] = rotated_spots[mapping]
        new_adata.uns['transformation'] = {
            'method': 'sequencing_rotation',
            'rotation_angle': rotation_angle,
            'center_correction': center_correction,
            'min_space': min_space,
            'original_grid_size': estimated_grid_size,
            'actual_grid_size': len(grid),
            'original_spots': len(self.spatial_data),
            'remaining_spots': len(mapping)
        }
        
        return new_adata
    
    def transform_imaging(self, rotation_angle=0, center_correction=0, keep_bounds=True):
        """
        Transform imaging-based spatial data with direct rotation.
        
        Parameters
        ----------
        rotation_angle : float
            Rotation angle in degrees
        center_correction : float or array-like
            Center point for rotation
        keep_bounds : bool
            If True, only keep spots that remain within the original spatial bounds
            
        Returns
        -------
        AnnData
            Transformed AnnData object with directly rotated coordinates
        """
        print(f"Transforming imaging-based data with {self.spatial_data.shape[0]} spots")
        
        # Get original bounds
        original_bounds = self._get_spatial_bounds(self.spatial_data)
        
        # Rotate the coordinates
        print(f"Rotating spots by {rotation_angle} degrees")
        rotated_spots = self._rotate(self.spatial_data, rotation_angle, center_correction)
        
        # Filter points if needed
        if keep_bounds:
            print("Filtering spots to keep only those within original bounds")
            in_bounds_mask = self._filter_points_in_range(rotated_spots, original_bounds)
            mapping = np.where(in_bounds_mask)[0]
            print(f"Keeping {len(mapping)} of {len(rotated_spots)} spots within bounds")
        else:
            mapping = np.arange(len(rotated_spots))
        
        # Create new AnnData object
        print(f"Creating new AnnData with {len(mapping)} spots")
        new_adata = self.adata[mapping, :].copy()
        new_adata.obsm['spatial'] = rotated_spots[mapping]
        new_adata.obsm['spatial_original'] = self.spatial_data[mapping]
        new_adata.uns['transformation'] = {
            'method': 'imaging_rotation',
            'rotation_angle': rotation_angle,
            'center_correction': center_correction,
            'kept_bounds': keep_bounds,
            'original_spots': len(self.spatial_data),
            'remaining_spots': len(mapping)
        }
        
        return new_adata


class WarpTransformer(SpatialTransformer):
    """Transform spatial data using thin plate spline warping."""
    
    def __init__(self, adata, distort_level=100, grid_size=3, alpha=1.0):
        super().__init__(adata)
        self.distort_level = distort_level
        self.grid_size = grid_size
        self.tps = ThinPlateSpline(alpha=alpha)
    
    def _create_grid(self, x_min, x_max, y_min, y_max):
        """Create a grid of control points with corners."""
        x = np.linspace(x_min, x_max, self.grid_size) + np.random.normal(0, 1, self.grid_size)
        y = np.linspace(y_min, y_max, self.grid_size) + np.random.normal(0, 1, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        
        margin = 0.05
        dx = (x_max - x_min) * margin
        dy = (y_max - y_min) * margin
        corner_points = np.array([
            [x_min + dx, y_min + dy],
            [x_min + dx, y_max - dy],
            [x_max - dx, y_min + dy],
            [x_max - dx, y_max - dy]
        ])
        
        return np.vstack((grid_points, corner_points))
    
    def _add_noise(self, points, is_corner=False):
        """Add noise to control points with different levels for grid vs corners."""
        if is_corner:
            return points + np.random.normal(0, 0.1, points.shape)
        
        h_grid = np.ptp(points[:, 1])
        w_grid = np.ptp(points[:, 0])
        variance = min(self.distort_level * np.sqrt(h_grid**2 + w_grid**2), 100)
        return points + np.random.normal(0, variance, points.shape)
    
    def _random_rotation_translation(self, points):
        """Apply a random rotation and translation to points."""
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        range_x = np.ptp(points[:, 0])
        range_y = np.ptp(points[:, 1])
        max_translation = min(range_x, range_y) * 0.1
        translation = np.random.uniform(-max_translation, max_translation, 2)
        
        return points @ rotation_matrix + translation
    
    def transform(self, apply_rotation=True):
        """Apply TPS warping transformation with optional rotation/translation."""
        try:
            x_min, x_max = self.spatial_data[:, 0].min(), self.spatial_data[:, 0].max()
            y_min, y_max = self.spatial_data[:, 1].min(), self.spatial_data[:, 1].max()
            source_points = self._create_grid(x_min, x_max, y_min, y_max)
            source_points = np.unique(source_points, axis=0)
            
            n_grid_points = self.grid_size * self.grid_size
            target_points = np.vstack((
                self._add_noise(source_points[:n_grid_points], False),
                self._add_noise(source_points[n_grid_points:], True)
            ))
            
            self.tps.fit(source_points, target_points)
            warped_coords = self.tps.transform(self.spatial_data)
            
            if apply_rotation:
                warped_coords = self._random_rotation_translation(warped_coords)
            
            warped_adata = self.adata.copy()
            warped_adata.obsm['spatial'] = warped_coords
            warped_adata.uns['transformation'] = {
                'method': 'TPS',
                'distort_level': self.distort_level,
                'grid_size': self.grid_size,
                'alpha': self.tps.alpha
            }
            
            return warped_adata
            
        except np.linalg.LinAlgError:
            print("Warning: TPS fitting failed, returning slightly perturbed data")
            warped_adata = self.adata.copy()
            perturbed_coords = self.spatial_data + np.random.normal(0, np.ptp(self.spatial_data)/100, self.spatial_data.shape)
            warped_adata.obsm['spatial'] = perturbed_coords
            return warped_adata