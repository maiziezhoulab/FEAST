from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
KEY_COLUMNS = ('method', 'mode', 'alpha', 'support_variant', 'representation')
ALL_SPOTS = 'all_spots_primary'
PAIRED_SUPPORT = 'paired_zero_panel_spots_excluded_sensitivity'
REPRESENTATIONS = {'GraphST': ('pca20_randomized_seed42', 'pca10_randomized_seed42_sensitivity'), 'STAMP': ('native_10d_topics',), 'scVI': ('native_10d_latent',)}

def alpha_text(value: object) -> str:
    return f'{float(value):.2f}'

def normalized_keys(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result['method'] = result['method'].astype(str)
    result['mode'] = result['mode'].astype(str)
    result['alpha'] = result['alpha'].map(alpha_text)
    result['support_variant'] = result['support_variant'].astype(str)
    result['representation'] = result['representation'].astype(str)
    return result

def expected_key_frame(*, modes: Iterable[str], alpha_levels: Iterable[object], common_support_cells: Iterable[tuple[str, float]]) -> pd.DataFrame:
    support_cells = {(str(mode), alpha_text(alpha)) for mode, alpha in common_support_cells}
    rows: list[dict[str, str]] = []
    for method, representations in REPRESENTATIONS.items():
        for mode in modes:
            for raw_alpha in alpha_levels:
                alpha = alpha_text(raw_alpha)
                if alpha == '0.00':
                    continue
                supports = [ALL_SPOTS]
                if (str(mode), alpha) in support_cells:
                    supports.append(PAIRED_SUPPORT)
                for support in supports:
                    for representation in representations:
                        rows.append({'method': method, 'mode': str(mode), 'alpha': alpha, 'support_variant': support, 'representation': representation})
    return pd.DataFrame(rows, columns=KEY_COLUMNS).sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

def assert_exact_key_matrix(table: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    missing_columns = set(KEY_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(f'{label} lacks key columns: {sorted(missing_columns)}')
    observed = normalized_keys(table)[list(KEY_COLUMNS)]
    if observed.duplicated().any():
        duplicate = observed.loc[observed.duplicated(keep=False)].head().to_dict('records')
        raise ValueError(f'{label} contains duplicate metric keys: {duplicate}')
    observed = observed.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    if not observed.equals(expected):
        merged = expected.merge(observed, on=list(KEY_COLUMNS), how='outer', indicator=True)
        mismatch = merged.loc[merged['_merge'] != 'both'].head(20).to_dict('records')
        raise ValueError(f'{label} does not match the expected support and representation keys: {mismatch}')
