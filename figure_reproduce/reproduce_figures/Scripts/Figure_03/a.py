"""Render clustering agreement across sections and signed logit shifts."""

from itertools import product

import numpy as np
import pandas as pd

from layout import CLUSTERING, HEADING_COLOR, PANELS, manuscript_axes, manuscript_text, panel, panel_args, save_panel, style_axis

METRICS = ("ARI", "NMI", "AMI")
SHIFTS = (-1, -0.5, -0.25, 0, 0.25, 0.5, 1)
SECTIONS = ("151508", "151670", "151676")
MEAN_LINESTYLES = {"GraphST": "-", "STAGATE": "--", "Leiden": "-."}


def source_data(study, plot_data=None):
    scores = pd.read_csv(study / "metrics/fixed_panel_benchmark_metrics.csv", dtype={"slice_id": str})
    keys = ["slice_id", "simulation_id"]
    lineage = ["feast_version", "public_seed"] if plot_data is None else []
    if plot_data is None:
        simulations = pd.read_csv(study / "run/simulation_manifest.csv", dtype={"slice_id": str})
        shifts = simulations.loc[
            simulations.alteration_type.eq("sparsity_logit_shift") | simulations.simulation_id.eq("baseline"),
            keys + ["alteration_value", "status", *lineage],
        ].rename(columns={"alteration_value": "logit_shift"})
    else:
        archived = pd.read_csv(plot_data, dtype={"slice_id": str})
        archived = archived.loc[archived.plot_group.eq("sparsity")]
        score_keys = keys + ["method"]
        compared = archived.merge(scores, on=score_keys, validate="one_to_one", suffixes=("_archive", ""))
        if len(compared) != len(archived):
            raise ValueError(f"Archived sparsity scores are missing from {study}")
        for metric in METRICS:
            np.testing.assert_allclose(compared[f"{metric}_archive"], compared[metric], rtol=0, atol=1e-12)
        shifts = pd.concat([
            archived[keys + ["level", "status"]].rename(columns={"level": "logit_shift"}),
            scores.loc[scores.simulation_id.eq("baseline"), keys + ["status"]],
        ]).drop_duplicates()
    shifts.loc[shifts.simulation_id.eq("baseline"), "logit_shift"] = 0
    if not shifts.status.eq("ok").all():
        raise ValueError("A plotted sparsity simulation did not complete")
    rows = scores.merge(shifts.drop(columns="status"), on=keys, validate="many_to_one",
                        suffixes=("_score", ""))
    for name in lineage:
        if f"{name}_score" in rows and not rows[f"{name}_score"].eq(rows[name]).all():
            raise ValueError(f"The plotted scores and simulations differ in {name}")
    expected = set(product(SECTIONS, SHIFTS, CLUSTERING))
    observed = set(zip(rows.slice_id, rows.logit_shift, rows.method))
    if len(rows) != len(expected) or observed != expected or not rows.status.eq("ok").all():
        raise ValueError("Expected 63 distinct successful sparsity scores")
    return rows[keys + ["logit_shift", "method", *METRICS, *lineage]]


def main():
    args = panel_args("a", __doc__, plot_data_help="Use archived alteration_plot_data.csv for the signed shift levels.")
    study = args.results / "01_clustering"
    rows = source_data(study, args.plot_data)
    means = rows.groupby(["method", "logit_shift"], as_index=False)[list(METRICS)].mean()
    tables = study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    rows.sort_values(["slice_id", "method", "logit_shift"]).to_csv(tables / "a_scores.csv", index=False)
    means.to_csv(tables / "a_section_means.csv", index=False)

    fig = panel("a")
    manuscript_text(fig, "a", 2.8, 28.4, "Clustering Benchmark", ha="left",
                    color=HEADING_COLOR, weight="bold", fontsize=11.8)
    manuscript_text(fig, "a", 89.5, 55.6, "Sparsity", ha="center", fontsize=8.6)
    axes = [manuscript_axes(fig, "a", 33.4, top, 104.6, 64.0) for top in (59.5, 144.7, 230.0)]
    for axis, metric in zip(axes, METRICS):
        for method, (name, color, marker) in CLUSTERING.items():
            for _, section in rows.loc[rows.method.eq(method)].groupby("slice_id"):
                section = section.sort_values("logit_shift")
                axis.plot(section.logit_shift, section[metric], color=color, alpha=0.32, lw=0.7)
            mean = means.loc[means.method.eq(method)].sort_values("logit_shift")
            axis.plot(mean.logit_shift, mean[metric], color=color, marker=marker, lw=1.45,
                      ls=MEAN_LINESTYLES[name],
                      ms=2.9, mec="white", mew=0.3, label=name)
        axis.axvline(0, color="#888888", lw=0.65, ls=(0, (3, 2)), zorder=0)
        style_axis(axis, max_ticks=4)
        axis.set_ylabel(f"{metric} ↑")
        axis.set_xlim(-1.12, 1.12)
        axis.set_xticks([-1, -0.5, 0, 0.5, 1], ["−1.0", "−0.5", "0.0", "0.5", "1.0"])
        axis.tick_params(axis="y", labelleft=False)
        axis.tick_params(axis="x", labelsize=5.8)
    manuscript_text(fig, "a", 94, 309, "Signed logit shift (neutral = 0)", ha="center", fontsize=6.0,
                    color=HEADING_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.56, 1 - 40.6 / PANELS["a"][3]),
               ncols=3, handlelength=1.4, columnspacing=0.7,
               prop={"size": 6.1, "weight": "bold"})
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
