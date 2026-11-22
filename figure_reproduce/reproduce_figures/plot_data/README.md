# Figure plotting inputs

These are the values and artwork read by the manuscript renderers. The `inputs/` and `results/` directories retain the relative paths of the study archive so the same panel code can read both. Raw expression matrices, fitted model objects and complete simulation runs belong in the external Zenodo study archive.

- **Figure 1** uses the author-supplied `Figures/Figure1_pipeline.pdf` as fixed artwork.
- **Figure 2b** uses the Study 00 visualization export `simulator_spatial_comparison.pdf` and its color limits from `visualization/00_simulator_benchmark/figures/`. The renderer extracts the 42 map tiles and draws the journal labels, scales and layout. The other Figure 2 inputs are numerical scores.
- **Figure 3b** uses section 151676. Each row of `Figure_03/b_partitions.csv.gz` contains a spot barcode, XY coordinates, a measured tissue layer, and the cluster labels for three methods at five sparsity shifts. Figure 3d uses the Study 03 archived spatial-composition export for section 100.
- **Figures 5–6** use the study’s plotting exports. Figure 5a retains both the full map context and the retained-spot mask. The Parquet projections preserve the plotted rows and column values from the marker, atlas-plane and section tables. Figure 6a uses the three archived anatomy images and the measured slice’s XY coordinates and region labels.
- **Figure 7b** uses eight archived native route images. The accompanying JSON supplies their crop rectangles and physical scale for the 1 mm bars.
- **Supplementary tables** retain the statistics, marker values, coordinates and labels needed for each plot. The S4 gene table retains the task and gene IDs, Pearson coefficient and finite-value flag. S5 uses saved expression-derived cluster assignments and embedding coordinates.

See the [render commands](../README.md) to rebuild the figures.
