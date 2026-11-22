"""Combine the retained article rows and the fresh FEAST-only Study 00 scores."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from itertools import product
from os.path import relpath
from pathlib import Path

from run import load_config

METRICS = "simulator_quality_metrics.csv"
MORAN = "moran_gene_panel.csv"
PAIR = ("simulator", "sample")
PROFILE_KEY = (*PAIR, "rank")
PROFILE_COLUMNS = (*PROFILE_KEY, "gene_id", "reference_moran_i", "simulated_moran_i")
COMPARATORS = {"SRTsim", "Splatter", "Splatter_Simple", "scCube"}
FEAST = "FEAST_reference_rank"


def open_table(path, mode):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8", newline="")


def read_table(path):
    with open_table(path, "rt") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def keyed_rows(rows, columns):
    keyed = {tuple(row[column] for column in columns): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate rows on {columns}")
    return keyed


def check_support(scores, profile, pairs, panel_size):
    if set(keyed_rows(scores, PAIR)) != pairs:
        raise ValueError("Score table does not cover the registered simulator/sample pairs")
    if any(int(row["n_moran_genes"]) != panel_size for row in scores):
        raise ValueError("Unexpected Moran gene count in the score table")

    keyed_rows(profile, PROFILE_KEY)
    groups = defaultdict(list)
    for row in profile:
        groups[tuple(row[column] for column in PAIR)].append(row)
    if set(groups) != pairs:
        raise ValueError("Moran table does not cover the registered simulator/sample pairs")
    for pair, rows in groups.items():
        ranks = sorted(int(row["rank"]) for row in rows)
        genes = {row["gene_id"] for row in rows}
        if ranks != list(range(1, panel_size + 1)) or len(genes) != panel_size or "" in genes:
            raise ValueError(f"Moran ranks or gene identifiers are incomplete for {pair}")


def assemble(config_path, retained_dir, feast_dir, output_dir):
    config = load_config(config_path)
    if set(config["external_simulators"]) != COMPARATORS or config["feast"]["simulator_name"] != FEAST:
        raise ValueError("Unexpected Study 00 simulator IDs")

    retained_scores = retained_dir / METRICS
    retained_profile = retained_dir / MORAN
    feast_scores = feast_dir / METRICS
    feast_profile = feast_dir / f"{MORAN}.gz"
    metric_columns, external_metrics = read_table(retained_scores)
    moran_columns, external_moran = read_table(retained_profile)
    new_metric_columns, feast_metrics = read_table(feast_scores)
    new_moran_columns, feast_moran = read_table(feast_profile)
    if not set(metric_columns) <= set(new_metric_columns):
        raise ValueError("The fresh scores are missing public metric fields")
    if set(moran_columns) != set(PROFILE_COLUMNS) or not set(moran_columns) <= set(new_moran_columns):
        raise ValueError("Unexpected Moran profile fields")

    samples = config["samples"]
    panel_size = config["metrics"]["max_moran_genes"]
    check_support(external_metrics, external_moran, set(product(COMPARATORS, samples)), panel_size)
    check_support(feast_metrics, feast_moran, set(product([FEAST], samples)), panel_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        (output_dir / METRICS, metric_columns, external_metrics, feast_metrics, PAIR),
        (output_dir / f"{MORAN}.gz", moran_columns, external_moran, feast_moran, PROFILE_KEY),
    ]
    for path, columns, retained, fresh, key in tables:
        # Copy the published fields verbatim, including missing cells and full precision.
        rows = [{column: row[column] for column in columns} for row in [*retained, *fresh]]
        with open_table(path, "wt") as stream:
            writer = csv.DictWriter(stream, columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        _, saved = read_table(path)
        if keyed_rows(saved, key) != keyed_rows(rows, key):
            raise ValueError(f"Written records differ from the source fields in {path}")

    provenance = {
        "configuration_id": config["configuration_id"],
        "public_seed": config["public_seed"],
        "paths_relative_to": "this provenance file's directory",
        "retained": {
            "simulators": sorted(COMPARATORS),
            "metrics": relpath(retained_scores, output_dir),
            "metric_rows": len(external_metrics),
            "moran": relpath(retained_profile, output_dir),
            "moran_rows": len(external_moran),
        },
        "fresh": {
            "simulator": FEAST,
            "metrics": relpath(feast_scores, output_dir),
            "metric_rows": len(feast_metrics),
            "moran": relpath(feast_profile, output_dir),
            "moran_rows": len(feast_moran),
            "scoring_provenance": relpath(feast_dir / "provenance.json", output_dir),
        },
        "output_rows": {
            METRICS: len(external_metrics) + len(feast_metrics),
            f"{MORAN}.gz": len(external_moran) + len(feast_moran),
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retained-dir", type=Path, required=True)
    parser.add_argument("--feast-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.config, args.retained_dir, args.feast_dir, args.output_dir)


if __name__ == "__main__":
    main()
