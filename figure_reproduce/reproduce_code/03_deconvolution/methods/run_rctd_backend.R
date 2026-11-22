#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(spacexr)
  library(SummarizedExperiment)
  library(SpatialExperiment)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) {
  position <- which(args == name)
  if (length(position) != 1 || position == length(args)) stop(paste("Missing", name))
  args[position + 1]
}

counts_file <- arg("--counts")
coords_file <- arg("--coords")
reference_file <- arg("--reference")
cell_types_file <- arg("--cell-types")
output_file <- arg("--output")
cache_dir <- arg("--cache-dir")
max_cores <- as.integer(arg("--max-cores"))
public_seed <- as.integer(arg("--seed"))
set.seed(public_seed)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

spatial_counts <- Matrix(as.matrix(read.csv(counts_file, row.names = 1, check.names = FALSE)), sparse = TRUE)
coordinates <- read.csv(coords_file, row.names = 1, check.names = FALSE)
reference_counts <- as.matrix(read.csv(reference_file, row.names = 1, check.names = FALSE))
storage.mode(reference_counts) <- "integer"
cell_types <- read.csv(cell_types_file, row.names = 1, check.names = FALSE)$cell_type
names(cell_types) <- colnames(reference_counts)

stopifnot(identical(rownames(spatial_counts), rownames(reference_counts)))
stopifnot(identical(colnames(spatial_counts), rownames(coordinates)))
stopifnot(identical(colnames(reference_counts), names(cell_types)))

reference_n_umi <- as.integer(colSums(reference_counts))
stopifnot(all(reference_n_umi > 0L))
reference <- SummarizedExperiment(
  assays = list(counts = reference_counts),
  colData = data.frame(
    cell_type = factor(cell_types),
    nUMI = reference_n_umi,
    row.names = colnames(reference_counts)
  )
)
spatial <- SpatialExperiment(
  assays = list(counts = spatial_counts),
  spatialCoords = as.matrix(coordinates[, c("x", "y")])
)
colData(spatial)$nUMI <- as.integer(coordinates$nUMI)

cat(sprintf("runtime::R=%s\n", R.version.string))
cat(sprintf("runtime::spacexr=%s\n", as.character(packageVersion("spacexr"))))
cat(sprintf("runtime::SummarizedExperiment=%s\n", as.character(packageVersion("SummarizedExperiment"))))
cat(sprintf("runtime::SpatialExperiment=%s\n", as.character(packageVersion("SpatialExperiment"))))
cat("Running createRctd\n")
fitted <- createRctd(spatial, reference, UMI_min = 1, UMI_min_sigma = 1)
cat("Running runRctd (full mode)\n")
result <- runRctd(fitted, rctd_mode = "full", max_cores = max_cores)
stopifnot("weights" %in% assayNames(result))
weights <- t(as.matrix(assay(result, "weights")))
row_totals <- rowSums(weights)
stopifnot(all(is.finite(weights)), all(row_totals > 0))
weights <- weights / row_totals
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(as.data.frame(weights), output_file, row.names = TRUE)
cat("Done.\n")
