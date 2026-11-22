"""Verify Study 00 sources and prepare Supplementary Figure S1 plot data."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

STUDY = "00_simulator_benchmark"
METHOD_NAMES = {"FEAST_reference_rank": "FEAST", "SRTsim": "SRTsim", "Splatter": "Splatter",
                "Splatter_Simple": "Splatter Simple", "scCube": "scCube"}
METRICS = {
    "input_mean_corr": ("correlation", "higher"),
    "input_variance_corr": ("correlation", "higher"),
    "relative_error_mean": ("log", "lower"),
    "moran_i_correlation": ("correlation", "higher"),
    "gene_zero_fraction_wasserstein": ("log", "lower"),
    "zero_mask_jaccard": ("unit", "higher"),
    "cosine_divergence": ("unit", "lower"),
    "library_size_wasserstein": ("log", "lower"),
}
EXAMPLES = (("b","DLPFC_151670",18326,3480), ("c","MERFISH_120",1122,22270))
VALUES = ["mean", "variance", "zero_fraction", "log1p_mean", "log1p_variance"]


def summary(data, genes):
    """Population moments on native X, as in the article's S1 renderer."""
    positions = data.var_names.get_indexer(genes)
    assert np.all(positions >= 0) and len(positions) == data.n_vars
    sums = np.zeros(len(genes), dtype=np.float64)
    squares = np.zeros(len(genes), dtype=np.float64)
    nonzeros = np.zeros(len(genes), dtype=np.int64)
    for start in range(0, data.n_obs, 2048):
        block = data.X[start:start+2048, :][:, positions]
        if sparse.issparse(block):
            block = block.tocsr().astype(np.float64)
            block.sum_duplicates()
            block.eliminate_zeros()
            values = block.data
            sums += np.asarray(block.sum(axis=0)).ravel()
            nonzeros += block.getnnz(axis=0)
            block.data **= 2
            squares += np.asarray(block.sum(axis=0)).ravel()
        else:
            block = np.asarray(block, dtype=np.float64)
            values = block
            sums += block.sum(axis=0)
            squares += np.square(block).sum(axis=0)
            nonzeros += np.count_nonzero(block, axis=0)
        assert np.isfinite(values).all() and (values >= 0).all()
    mean = sums / data.n_obs
    variance = np.maximum(squares / data.n_obs - mean ** 2, 0)
    return pd.DataFrame({"gene_id": genes, "n_spots": data.n_obs, "mean": mean,
                         "variance": variance, "zero_fraction": 1-nonzeros/data.n_obs,
                         "log1p_mean": np.log1p(mean), "log1p_variance": np.log1p(variance)})


def prepare(zenodo):
    inputs, results = (zenodo / kind / STUDY for kind in ("inputs", "results"))
    output = results / "s1"
    output.mkdir(parents=True, exist_ok=True)
    score_path = results / "metrics/simulator_quality_metrics.csv"
    metrics = pd.read_csv(score_path)
    samples = yaml.safe_load((inputs / "config.yaml").read_text())["samples"]
    assert len(metrics) == 60 and set(metrics.simulator) == set(METHOD_NAMES)
    assert set(zip(metrics["sample"], metrics.simulator)) == {(sample, method) for sample in samples for method in METHOD_NAMES}
    long = metrics.assign(method=metrics.simulator.map(METHOD_NAMES)).melt(
        id_vars=["sample", "simulator", "method", "n_common_genes", "n_paired_spots", "spot_pairing", "n_moran_genes"],
        value_vars=list(METRICS), var_name="metric", value_name="value").assign(panel_id="a")
    long["direction"] = long.metric.map({metric: spec[1] for metric,spec in METRICS.items()})
    key = ["sample", "method", "metric"]
    shared = pd.read_csv(results / "figure2/plot_data_long.csv").rename(columns={"metric":"label", "source_column":"metric"})
    fresh = long.set_index(key, verify_integrity=True).value
    pd.testing.assert_series_equal(fresh.sort_index(), shared.set_index(key, verify_integrity=True).score.sort_index(),
                                   check_names=False, check_exact=True)
    article = pd.read_csv(results / "article/s1/Fig_S1_metrics.csv")
    previous = article.set_index(key, verify_integrity=True).value
    pd.testing.assert_index_equal(fresh.sort_index().index, previous.sort_index().index)
    benchmark = fresh.rename("current_value").to_frame().join(previous.rename("article_value"))
    benchmark["delta"] = benchmark.current_value - benchmark.article_value
    comparator = benchmark.index.get_level_values("method") != "FEAST"
    np.testing.assert_allclose(benchmark.loc[comparator,"delta"],0,rtol=0,atol=1e-12)
    scales = {}
    for metric, (kind, direction) in METRICS.items():
        values = metrics[metric]
        assert np.isfinite(values).all()
        if kind == "log":
            assert values.min() > 0
            limits = [float(10**np.floor(np.log10(values.min()))), float(10**np.ceil(np.log10(values.max())))]
        else:
            limits = [-1 if kind == "correlation" else 0, 1]
            assert values.between(*limits).all()
        scales[metric] = {"kind": kind, "direction": direction, "limits": limits}
    archive = pd.read_csv(results / "article/s1/Fig_S1_clouds.csv.gz", float_precision="round_trip",
                          dtype={"gene_id":str})
    output_tables = []
    dimensions = []
    statistic_ranges = {}
    for letter, sample, n_genes, n_spots in EXAMPLES:
        filename = f"{sample}.h5ad"
        reference = ad.read_h5ad(inputs / "reference" / filename)
        genes = reference.var_names
        assert reference.shape == (n_spots, n_genes) and genes.is_unique and reference.obs_names.is_unique
        paths = {"Reference": inputs / "reference" / filename,
                 "FEAST": results / "run/simulations" / filename}
        paths.update({method:inputs / "external" / simulator / filename
                      for simulator, method in METHOD_NAMES.items() if method != "FEAST"})
        group = []
        for source, path in paths.items():
            data = reference if source == "Reference" else ad.read_h5ad(path)
            assert data.shape == reference.shape and data.obs_names.is_unique and data.var_names.is_unique
            if source == "FEAST":
                assert data.obs_names.equals(reference.obs_names)
            computed = summary(data, genes).set_index("gene_id")
            prior = archive.loc[archive["sample"].eq(sample) & archive.source.eq(source)].set_index("gene_id", verify_integrity=True)
            assert len(prior) == n_genes and set(prior.index) == set(genes)
            assert prior.n_spots.eq(n_spots).all()
            prior = prior.loc[genes]
            difference = (computed[VALUES] - prior[VALUES]).abs().max()
            if source != "FEAST":
                np.testing.assert_allclose(computed[VALUES], prior[VALUES], rtol=1e-10, atol=1e-10, err_msg=str(path))
            table = computed if source == "FEAST" else prior[["n_spots", *VALUES]].copy()
            table = table.rename_axis("gene_id").reset_index()
            table = table.assign(panel_id=letter, sample=sample, source=source, source_path=str(path.relative_to(zenodo)),
                                 archived_source_path=prior.source_path.iloc[0])
            group.append(table)
            dimensions.append({"panel_id":letter, "sample":sample, "source":source, "spots":n_spots,
                               "genes":n_genes, "zero_mean_genes":int(table["mean"].eq(0).sum()),
                               "archived_zero_mean_genes":int(prior["mean"].eq(0).sum()),
                               "same_spot_order_as_reference":data.obs_names.equals(reference.obs_names),
                               "maximum_absolute_statistic_delta":difference.to_dict(),
                               "h5ad":str(path.relative_to(zenodo))})
        frame = pd.concat(group, ignore_index=True)
        ranges = {}
        for column in ("log1p_mean", "log1p_variance"):
            maximum = float(frame[column].max())
            ranges[column] = [-.025*maximum, 1.04*maximum]
        statistic_ranges[sample] = ranges
        output_tables.append(frame)
    clouds = pd.concat(output_tables, ignore_index=True)
    assert clouds.shape[0] == 116688 and not clouds.duplicated(["sample","source","gene_id"]).any()
    long.to_csv(output / "Fig_S1_metrics.csv", index=False)
    clouds.to_csv(output / "Fig_S1_clouds.csv.gz", index=False)
    benchmark.reset_index().to_csv(output / "benchmark_comparison.csv", index=False)
    metadata = {"sample_order":samples, "score_rows":int(len(metrics)), "heatmap_values":int(len(long)),
                "cloud_rows":int(len(clouds)), "metric_scales":scales, "cloud_limits":statistic_ranges,
                "zero_fraction_color_limits":[0,1], "cloud_sources":dimensions,
                "archived_comparator_values_verified":int(comparator.sum()),
                "corrected_scCube_Slide_seq_Moran":float(metrics.loc[metrics["sample"].eq("Slideseq_001")
                    & metrics.simulator.eq("scCube"),"moran_i_correlation"].item())}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=Path(__file__).resolve().parents[2]/"Zenodo")
    prepare(parser.parse_args().zenodo)
