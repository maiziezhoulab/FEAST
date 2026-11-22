#!/usr/bin/env bash
# Render the journal PDFs using the plotting inputs distributed with FEAST.
set -eu
cd "$(dirname "$0")"
python_bin=${PYTHON:-python}
data=plot_data
results=$data/results
benchmark=$results/00_simulator_benchmark/article
if [[ $# == 0 ]]; then
    set -- 01 02 03 04 05 06 07 S1 S2 S3 S4 S5
fi
for figure in "$@"; do
    printf 'Building Figure %s\n' "$figure"
    out="Outputs/Figure_$figure"
    scripts="Scripts/Figure_$figure"
    if [[ $figure == 01 ]]; then
        mkdir -p "$out"
        cp "$data/Figure_01/Figure1_pipeline.pdf" "$out/"
        continue
    fi
    for script in "$scripts"/[a-f].py; do
        letter=${script##*/}
        case "$figure/$letter" in
            02/a.py) args=(--metrics-csv "$benchmark/metrics/simulator_quality_metrics.csv" --results-dir "$out/tables" --output-dir "$out") ;;
            02/b.py) args=(--archived-dir "$benchmark/figure2" --archived-pdf "$benchmark/figure2/simulator_spatial_comparison.pdf" --output-dir "$out") ;;
            03/*) args=(--results "$results" --tables-dir "$out/tables")
                  case "$letter" in
                      b.py) args+=(--plot-data "$data/Figure_03/b_partitions.csv.gz") ;;
                      d.py) args+=(--plot-data "$results/03_deconvolution/figure3/deconvolution_spatial_plot_data.csv.gz") ;;
                  esac ;;
            04/*) args=(--results "$results/04_batch_effect_removal")
                  if [[ $letter == c.py ]]; then args+=(--tables-dir "$out/tables"); fi ;;
            S*/*) args=(--zenodo "$data" --pdf-only) ;;
            *) args=(--zenodo "$data") ;;
        esac
        "$python_bin" "$script" "${args[@]}"
    done
    if [[ $figure == S* ]]; then
        "$python_bin" "$scripts/assemble.py" --pdf-only
    else
        "$python_bin" "$scripts/assemble.py"
    fi
done
