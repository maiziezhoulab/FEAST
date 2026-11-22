"""Read Study 03 paths relative to the manifest's run directory."""

import os

import pandas as pd

DATA_PATHS = (
    "reference_path", "simulation_path", "truth_path", "cell_level_path",
    "cell_assignment_path", "output_path", "metadata_path", "runner_log_path",
)


def read_manifest(path):
    table = pd.read_csv(path, dtype={"slice_id": str})
    for column in set(DATA_PATHS).intersection(table):
        table[column] = table[column].map(
            lambda value: str((path.parent / str(value)).resolve()) if pd.notna(value) else value
        )
    return table


def write_manifest(table, path):
    table = table.copy()
    for column in set(DATA_PATHS).intersection(table):
        table[column] = table[column].map(
            lambda value: os.path.relpath(value, path.parent) if isinstance(value, str) and value else value
        )
    table.to_csv(path, index=False)
