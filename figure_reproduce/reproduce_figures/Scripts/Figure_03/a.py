"""Render clustering agreement across sections and signed logit shifts."""

from itertools import product

import pandas as pd

from layout import CLUSTERING, panel, panel_args, save_panel, style_axis

METRICS = ("ARI", "NMI", "AMI")
SHIFTS = (-1, -0.5, -0.25, 0, 0.25, 0.5, 1)
SECTIONS = ("151508", "151670", "151676")


def source_data(study):
    scores = pd.read_csv(study / "metrics/fixed_panel_benchmark_metrics.csv", dtype={"slice_id": str})
    simulations = pd.read_csv(study / "run/simulation_manifest.csv", dtype={"slice_id": str})
    keys = ["slice_id", "simulation_id"]
    shifts = simulations.loc[
        simulations.alteration_type.eq("sparsity_logit_shift") | simulations.simulation_id.eq("baseline"),
        keys + ["alteration_value", "status", "feast_version", "public_seed"],
    ].rename(columns={"alteration_value": "logit_shift"})
    shifts.loc[shifts.simulation_id.eq("baseline"), "logit_shift"] = 0
    if not shifts.status.eq("ok").all():
        raise ValueError("A plotted sparsity simulation did not complete")
    rows = scores.merge(shifts.drop(columns="status"), on=keys, validate="many_to_one",
                        suffixes=("_score", ""))
    for name in ("feast_version", "public_seed"):
        if f"{name}_score" in rows and not rows[f"{name}_score"].eq(rows[name]).all():
            raise ValueError(f"The plotted scores and simulations differ in {name}")
    expected = set(product(SECTIONS, SHIFTS, CLUSTERING))
    observed = set(zip(rows.slice_id, rows.logit_shift, rows.method))
    if len(rows) != len(expected) or observed != expected or not rows.status.eq("ok").all():
        raise ValueError("Expected 63 distinct successful sparsity scores")
    return rows[keys + ["logit_shift", "method", *METRICS, "feast_version", "public_seed"]]


def main():
    args = panel_args("a", __doc__)
    study = args.results / "01_clustering"
    rows = source_data(study)
    means = rows.groupby(["method", "logit_shift"], as_index=False)[list(METRICS)].mean()
    tables = study / "figure3"
    tables.mkdir(parents=True, exist_ok=True)
    rows.sort_values(["slice_id", "method", "logit_shift"]).to_csv(tables / "a_scores.csv", index=False)
    means.to_csv(tables / "a_section_means.csv", index=False)

    fig = panel("a")
    fig.text(0.51, 0.925, "Clustering benchmark", ha="center", weight="bold", fontsize=11)
    axes = fig.subplots(3, 1, sharex=True)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.811, bottom=0.10, hspace=0.28)
    for axis, metric in zip(axes, METRICS):
        for method, (name, color, marker) in CLUSTERING.items():
            for _, section in rows.loc[rows.method.eq(method)].groupby("slice_id"):
                section = section.sort_values("logit_shift")
                axis.plot(section.logit_shift, section[metric], color=color, alpha=0.32, lw=0.7)
            mean = means.loc[means.method.eq(method)].sort_values("logit_shift")
            axis.plot(mean.logit_shift, mean[metric], color=color, marker=marker, lw=1.45,
                      ms=2.9, mec="white", mew=0.3, label=name)
        axis.axvline(0, color="#888888", lw=0.65, ls=(0, (3, 2)), zorder=0)
        style_axis(axis, max_ticks=4)
        axis.set_ylabel(f"{metric} ↑")
        axis.set_xlim(-1.12, 1.12)
        axis.set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[-1].set_xlabel("Signed logit shift (neutral = 0)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.907),
               ncols=3, fontsize=6.4, handlelength=1.4, columnspacing=0.7)
    save_panel(fig, args.output)


if __name__ == "__main__":
    main()
