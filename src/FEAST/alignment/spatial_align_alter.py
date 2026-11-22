import numpy as np
import scipy.spatial
import anndata as ad
from tps import ThinPlateSpline


def rigid_rotation_transform(angle_degrees, center):
    """Return forward and inverse 2D homogeneous rotation matrices.

    Positive angles rotate counter-clockwise around ``center``.  The matrices
    act on homogeneous column vectors ``[x, y, 1]``.
    """
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must contain two finite coordinates")

    theta = np.deg2rad(float(angle_degrees))
    if not np.isfinite(theta):
        raise ValueError("angle_degrees must be finite")
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )

    forward = np.eye(3, dtype=np.float64)
    forward[:2, :2] = rotation
    forward[:2, 2] = center - rotation @ center
    inverse = np.linalg.inv(forward)
    return forward, inverse


def apply_spatial_transform(coords, transform):
    """Apply a 3x3 homogeneous transform to finite 2D coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_spots, 2)")
    if transform.shape != (3, 3):
        raise ValueError("transform must have shape (3, 3)")
    if not np.isfinite(coords).all() or not np.isfinite(transform).all():
        raise ValueError("coordinates and transform must be finite")

    homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
    transformed = homogeneous @ transform.T
    if not np.allclose(transformed[:, 2], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("transform produced invalid homogeneous coordinates")
    return transformed[:, :2]


def rotate_spatial(
    adata,
    angle_degrees,
    *,
    center=None,
    plate_bounds=None,
    spatial_key="spatial",
    original_key="spatial_original",
):
    """Rotate an AnnData slice, optionally cropping to a fixed plate.

    The transformation is centered on the coordinate centroid unless an
    explicit center is supplied.  By default it never changes spot support.
    Pass ``plate_bounds`` as ``[[x_min, y_min], [x_max, y_max]]`` to retain
    only rotated spots inside that inclusive rectangle.  It never snaps,
    deduplicates, or reorders spots.  Transform and crop metadata are recorded
    in ``.uns``.
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"obsm[{spatial_key!r}] is required")
    if not adata.obs_names.is_unique:
        raise ValueError("stable observation identifiers must be unique")

    spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(f"obsm[{spatial_key!r}] must have shape (n_spots, 2)")
    if spatial.shape[0] != adata.n_obs or not np.isfinite(spatial).all():
        raise ValueError("spatial coordinates must be finite and match n_obs")

    validated_bounds = None
    if plate_bounds is not None:
        try:
            validated_bounds = np.asarray(plate_bounds, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "plate_bounds must be finite [[x_min, y_min], [x_max, y_max]]"
            ) from error
        if (
            validated_bounds.shape != (2, 2)
            or not np.isfinite(validated_bounds).all()
        ):
            raise ValueError(
                "plate_bounds must be finite [[x_min, y_min], [x_max, y_max]]"
            )
        if not np.all(validated_bounds[0] < validated_bounds[1]):
            raise ValueError(
                "plate_bounds lower coordinates must be less than upper coordinates"
            )

    rotation_center = spatial.mean(axis=0) if center is None else np.asarray(center)
    forward, inverse = rigid_rotation_transform(angle_degrees, rotation_center)
    rotated = apply_spatial_transform(spatial, forward)

    retained_mask = np.ones(adata.n_obs, dtype=bool)
    if validated_bounds is not None:
        retained_mask = np.all(
            (rotated >= validated_bounds[0]) & (rotated <= validated_bounds[1]),
            axis=1,
        )
    result = (
        adata[retained_mask].copy()
        if validated_bounds is not None
        else adata.copy()
    )
    result.obsm[original_key] = spatial[retained_mask].copy()
    result.obsm[spatial_key] = rotated[retained_mask].copy()

    retained_n_spots = int(retained_mask.sum())
    transform_metadata = {
        "schema_version": 1,
        "method": "centered_rigid_rotation",
        "angle_degrees": float(angle_degrees),
        "center": rotation_center.astype(float),
        "forward_matrix": forward,
        "inverse_matrix": inverse,
        "spatial_key": spatial_key,
        "original_key": original_key,
        "preserve_spot_identity": True,
        "preserve_spot_count": retained_n_spots == adata.n_obs,
        "plate_bounds_applied": validated_bounds is not None,
        "input_n_spots": int(adata.n_obs),
        "retained_n_spots": retained_n_spots,
        "dropped_n_spots": int(adata.n_obs - retained_n_spots),
        "retained_fraction": float(retained_n_spots / adata.n_obs)
        if adata.n_obs
        else 1.0,
        "n_spots": int(adata.n_obs),
    }
    if validated_bounds is not None:
        transform_metadata["plate_bounds"] = validated_bounds.copy()
    result.uns["feast_alignment_transform"] = transform_metadata

    expected_names = adata.obs_names[retained_mask]
    if result.n_obs != retained_n_spots or not result.obs_names.equals(expected_names):
        raise RuntimeError("rotation changed expected spot support or ordering")
    return result

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
    
    def cut(self, x_min, x_max, y_min, y_max):
        """Cut a rectangular section from the spatial data."""
        mask = (
            (self.spatial_data[:, 0] >= x_min) & (self.spatial_data[:, 0] <= x_max) &
            (self.spatial_data[:, 1] >= y_min) & (self.spatial_data[:, 1] <= y_max)
        )
        adata_cut = self.adata[mask].copy()
        return adata_cut
    
    def move(self, dx, dy):
        x_min, x_max = np.min(self.adata.obsm['spatial'][:, 0]), np.max(self.adata.obsm['spatial'][:, 0])
        y_min, y_max = np.min(self.adata.obsm['spatial'][:, 1]), np.max(self.adata.obsm['spatial'][:, 1])
        
        moved_adata = self.adata.copy()
        moved_coords = self.spatial_data + np.array([dx, dy])
        
        in_bounds = (
            (moved_coords[:, 0] >= x_min) & (moved_coords[:, 0] <= x_max) &
            (moved_coords[:, 1] >= y_min) & (moved_coords[:, 1] <= y_max)
        )
        
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

    def transform_rigid(self, rotation_angle=0, center=None):
        """Apply the publication-safe centered continuous rotation."""
        return rotate_spatial(
            self.adata,
            rotation_angle,
            center=center,
        )
    
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
                             min_space=None):
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
        
        x_coords = self.spatial_data[:, 0]
        y_coords = self.spatial_data[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        x_range = [min_x, max_x]
        y_range = [min_y, max_y]
        
        if min_space is None:
            from scipy.spatial import distance_matrix
            
            sample_size = min(1000, len(self.spatial_data))
            indices = np.random.choice(len(self.spatial_data), size=sample_size, replace=False)
            sampled_coords = self.spatial_data[indices]
            
            dist_matrix = distance_matrix(sampled_coords, sampled_coords)
            np.fill_diagonal(dist_matrix, np.inf)
            min_space = np.min(dist_matrix)
            print(f"Calculated minimal spacing: {min_space}")
        
        spacing = min_space / 2
        offset = min_space * np.sqrt(3) / 2
        
        rows_full = round((y_range[1] - y_range[0]) / offset) + 1
        cols_full = round((x_range[1] - x_range[0]) / (spacing * 2)) + 1
        estimated_grid_size = rows_full * cols_full
        

        rows, cols = rows_full, cols_full
        
        print(f"Generating grid with {rows}x{cols}={rows*cols} points")
        grid = self._generate_offset_grid(rows, cols, spacing, offset, x_range, y_range)
        
        print(f"Rotating spots by {rotation_angle} degrees")
        rotated_spots = self._rotate(self.spatial_data, rotation_angle, center_correction)
        
        print(f"Finding nearest grid points")
        nearest_indices = self._find_nearest_grid_points(rotated_spots, grid)
        new_spots = grid[nearest_indices]
        
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
        
        original_bounds = self._get_spatial_bounds(self.spatial_data)
        
        print(f"Rotating spots by {rotation_angle} degrees")
        rotated_spots = self._rotate(self.spatial_data, rotation_angle, center_correction)
        
        if keep_bounds:
            print("Filtering spots to keep only those within original bounds")
            in_bounds_mask = self._filter_points_in_range(rotated_spots, original_bounds)
            mapping = np.where(in_bounds_mask)[0]
            print(f"Keeping {len(mapping)} of {len(rotated_spots)} spots within bounds")
        else:
            mapping = np.arange(len(rotated_spots))
        
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
            raise np.linalg.LinAlgError(
                "TPS fitting failed: the control point configuration is degenerate. "
                "Try reducing distort_level, increasing grid_size, or using different spatial data."
            )
