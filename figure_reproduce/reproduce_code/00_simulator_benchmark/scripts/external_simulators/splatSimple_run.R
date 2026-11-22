#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(optparse)
    library(splatter)
    library(Matrix)
    library(SummarizedExperiment)
})

option_list <- list(
    make_option(c("-c", "--cellinfo"), type = "character", default = NULL),
    make_option(c("-g", "--geneinfo"), type = "character", default = NULL),
    make_option(c("-m", "--matrix"), type = "character", default = NULL),
    make_option(c("-s", "--spatial_loc"), type = "character", default = NULL),
    make_option(c("-o", "--output_dir"), type = "character", default = "./")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$cellinfo) || is.null(opt$geneinfo) ||
    is.null(opt$matrix) || is.null(opt$spatial_loc)) {
    stop("All input files must be specified")
}

dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

read_spatial <- function(path, n_cells) {
    spatial <- read.csv(path, row.names = 1, check.names = FALSE)
    if (nrow(spatial) != n_cells) {
        spatial <- read.csv(path, check.names = FALSE)
    }
    if (!all(c("x", "y") %in% colnames(spatial))) {
        numeric_cols <- names(spatial)[vapply(spatial, is.numeric, logical(1))]
        if (length(numeric_cols) < 2) {
            stop("spatial.csv must contain x/y or at least two numeric columns")
        }
        spatial$x <- spatial[[numeric_cols[1]]]
        spatial$y <- spatial[[numeric_cols[2]]]
    }
    spatial
}

tryCatch({
    cat("Simple: Reading input files...\n")
    cellinfo <- read.csv(opt$cellinfo, row.names = 1, check.names = FALSE)
    geneinfo <- read.csv(opt$geneinfo, row.names = 1, check.names = FALSE)
    counts_in <- as(readMM(opt$matrix), "dgCMatrix")
    rownames(counts_in) <- rownames(geneinfo)
    colnames(counts_in) <- rownames(cellinfo)

    cat("Simple: Simulating counts with default Splatter parameters...\n")
    sim <- splatSimulate(
        nGenes = nrow(counts_in),
        batchCells = ncol(counts_in),
        verbose = FALSE
    )

    cat("Simple: Extracting simulated data and integrating spatial locations...\n")
    sim_counts <- as(assay(sim, "counts"), "dgCMatrix")
    if (nrow(sim_counts) != nrow(counts_in)) {
        stop("Simulated gene count does not match input gene count")
    }
    if (ncol(sim_counts) != ncol(counts_in)) {
        sim_counts <- sim_counts[, seq_len(min(ncol(sim_counts), ncol(counts_in))), drop = FALSE]
    }

    rownames(sim_counts) <- rownames(counts_in)
    colnames(sim_counts) <- colnames(counts_in)[seq_len(ncol(sim_counts))]

    spatial <- read_spatial(opt$spatial_loc, ncol(sim_counts))
    loc <- data.frame(
        x = spatial$x[seq_len(ncol(sim_counts))],
        y = spatial$y[seq_len(ncol(sim_counts))],
        label = colnames(sim_counts)
    )

    cat("Simple: Saving intermediate files to:", opt$output_dir, "\n")
    writeMM(sim_counts, file.path(opt$output_dir, "splatSimple_count.mtx"))
    write.csv(data.frame(gene = rownames(sim_counts)),
              file.path(opt$output_dir, "splatSimple_genes.csv"),
              row.names = FALSE)
    write.csv(loc, file.path(opt$output_dir, "splatSimple_loc.csv"), row.names = FALSE)
}, error = function(e) {
    cat("Simple error:\n")
    cat(conditionMessage(e), "\n")
    quit(status = 1)
})

