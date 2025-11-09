"""
Dataset path utilities for FEAST-sim reproduction scripts.

This module provides standardized dataset naming and path resolution
for all benchmarking scripts. Place your datasets in a 'data/' directory
following the naming convention specified in README.md.

Naming Convention:
    DLPFC_{sample_id}.h5ad          # e.g., DLPFC_151670.h5ad
    MERFISH_{slice_id}.h5ad         # e.g., MERFISH_007.h5ad
    OpenST_{slice_id}.h5ad          # e.g., OpenST_005.h5ad
    Stereoseq_{sample_id}.h5ad      # e.g., Stereoseq_E14_5_E2S2.h5ad
    Slideseq_{sample_id}.h5ad       # e.g., Slideseq_001.h5ad
    Xenium_{sample_id}.h5ad         # e.g., Xenium_LymphNode.h5ad
"""

from pathlib import Path
from typing import Union, List

# Repository root is 2 levels up from this file
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / 'data'


def get_data_path(dataset: str, sample_id: str) -> Path:
    """
    Get standardized path to a dataset file.
    
    Args:
        dataset: Dataset type ('DLPFC', 'MERFISH', 'OpenST', 'Stereoseq', 'Slideseq', 'Xenium')
        sample_id: Sample identifier (e.g., '151670', '007', 'E14_5_E2S2')
    
    Returns:
        Path to the dataset file
    
    Example:
        >>> get_data_path('DLPFC', '151670')
        PosixPath('/path/to/FEAST-sim/data/DLPFC_151670.h5ad')
    """
    filename = f"{dataset}_{sample_id}.h5ad"
    return DATA_DIR / filename


def get_merfish_slices(start: int, end: int) -> List[Path]:
    """
    Get multiple consecutive MERFISH slices.
    
    Args:
        start: Starting slice number
        end: Ending slice number (inclusive)
    
    Returns:
        List of paths to MERFISH slice files
    
    Example:
        >>> get_merfish_slices(5, 9)
        [.../MERFISH_005.h5ad, .../MERFISH_006.h5ad, ..., .../MERFISH_009.h5ad]
    """
    return [get_data_path('MERFISH', f'{i:03d}') for i in range(start, end + 1)]


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def check_dataset_exists(path: Union[Path, str], raise_error: bool = True) -> bool:
    """
    Check if a dataset file exists.
    
    Args:
        path: Path to check
        raise_error: If True, raise FileNotFoundError if file doesn't exist
    
    Returns:
        True if file exists, False otherwise
    """
    path = Path(path)
    if not path.exists():
        if raise_error:
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Please download datasets and place them in {DATA_DIR}/\n"
                f"See README.md for dataset sources and naming convention."
            )
        return False
    return True


# Predefined dataset shortcuts for common benchmarks
DATASETS = {
    # Simulator benchmarks
    'sim_dlpfc': get_data_path('DLPFC', '151670'),
    'sim_merfish_006': get_data_path('MERFISH', '006'),
    'sim_merfish_007': get_data_path('MERFISH', '007'),
    'sim_openst_005': get_data_path('OpenST', '005'),
    'sim_openst_006': get_data_path('OpenST', '006'),
    'sim_stereoseq': get_data_path('Stereoseq', 'E14_5_E2S2'),
    'sim_slideseq': get_data_path('Slideseq', '001'),
    'sim_xenium': get_data_path('Xenium', 'LymphNode'),
    
    # Clustering benchmarks
    'clust_dlpfc': get_data_path('DLPFC', '151676'),
    
    # Alignment benchmarks
    'align_dlpfc': get_data_path('DLPFC', '151675'),
    
    # Deconvolution benchmarks
    'deconv_merfish': get_data_path('MERFISH', '007'),
    
    # Interpolation benchmarks (returns list)
    'interp_merfish': get_merfish_slices(5, 9),
}


if __name__ == '__main__':
    # Print dataset paths for verification
    print("FEAST-sim Dataset Paths")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Repository root: {REPO_ROOT}")
    print("\nExpected dataset locations:")
    print("-" * 60)
    
    for name, path in DATASETS.items():
        if isinstance(path, list):
            print(f"\n{name}:")
            for p in path:
                status = "✓" if p.exists() else "✗"
                print(f"  {status} {p}")
        else:
            status = "✓" if path.exists() else "✗"
            print(f"{status} {name:20s} → {path}")
    
    print("\n" + "=" * 60)
    missing = sum(1 for p in DATASETS.values() 
                  if (isinstance(p, Path) and not p.exists()) or 
                     (isinstance(p, list) and any(not x.exists() for x in p)))
    if missing > 0:
        print(f"⚠ {missing} dataset(s) missing. See README.md for download instructions.")
    else:
        print("✓ All datasets found!")
