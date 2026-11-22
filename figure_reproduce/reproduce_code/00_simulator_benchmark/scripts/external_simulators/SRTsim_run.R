#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(optparse)
    library(SRTsim)
    library(Matrix)
})

option_list <- list(
    make_option(c("-c", "--cellinfo"), type = "character", default = NULL,
                help = "Path to cellinfo.csv"),
    make_option(c("-g", "--geneinfo"), type = "character", default = NULL,
                help = "Path to geneinfo.csv"),
    make_option(c("-m", "--matrix"), type = "character", default = NULL,
                help = "Path to sparse_matrix.mtx"),
    make_option(c("-s", "--spatial_loc"), type = "character", default = NULL,
                help = "Path to spatial.csv"),
    make_option(c("-o", "--output_dir"), type = "character", default = "./",
                help = "Output directory [default= %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$cellinfo) || is.null(opt$geneinfo) ||
    is.null(opt$matrix) || is.null(opt$spatial_loc)) {
    stop("All input files must be specified")
}

dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

process_spatial_data <- function(cellinfo_path, geneinfo_path, matrix_path, loc_path, output_dir) {
    tryCatch({
        cat("Reading input files...\n")
        ST_cellinfo <- read.csv(cellinfo_path, row.names = 1, check.names = FALSE)
        ST_geneinfo <- read.csv(geneinfo_path, row.names = 1, check.names = FALSE)
        ST_counts <- readMM(matrix_path)
        ST_counts <- as(ST_counts, "dgCMatrix")
        spatial_loc <- read.csv(loc_path, row.names = 1, check.names = FALSE)

        cat("Setting matrix dimensions...\n")
        rownames(ST_counts) <- rownames(ST_geneinfo)
        colnames(ST_counts) <- rownames(ST_cellinfo)

        if (nrow(spatial_loc) != ncol(ST_counts)) {
            spatial_loc <- read.csv(loc_path, check.names = FALSE)
        }
        if (!all(c("x", "y") %in% colnames(spatial_loc))) {
            numeric_cols <- names(spatial_loc)[vapply(spatial_loc, is.numeric, logical(1))]
            if (length(numeric_cols) < 2) {
                stop("spatial.csv must contain x/y or at least two numeric columns")
            }
            spatial_loc$x <- spatial_loc[[numeric_cols[1]]]
            spatial_loc$y <- spatial_loc[[numeric_cols[2]]]
        }

        cat("Creating location dataframe...\n")
        ST_loc <- data.frame(
            x = spatial_loc$x,
            y = spatial_loc$y,
            label = rownames(ST_cellinfo)
        )
        rownames(ST_loc) <- colnames(ST_counts)

        cat("Creating SRT object...\n")
        simSRT <- createSRT(count_in = ST_counts, loc_in = ST_loc)
        simSRT1 <- srtsim_fit(simSRT, sim_schem = "tissue")
        simSRT1 <- srtsim_count(simSRT1)

        cat("Extracting data from SRT object...\n")
        simSRT_count <- simCounts(simSRT1)
        simSRT_loc <- slot(simSRT, "refcolData")
        simSRT_loc_df <- as.data.frame(simSRT_loc)

        cat("Saving results...\n")
        writeMM(simSRT_count, file.path(output_dir, "SRTsim_count.mtx"))
        write.csv(simSRT_loc_df, file.path(output_dir, "SRTsim_loc.csv"), row.names = FALSE)
        write.csv(
            data.frame(gene = rownames(simSRT_count)),
            file.path(output_dir, "SRTsim_genes.csv"),
            row.names = FALSE
        )

        cat("Processing completed successfully\n")
        cat("Results saved in:", output_dir, "\n")
    }, error = function(e) {
        cat("Error occurred during processing:\n")
        cat(conditionMessage(e), "\n")
        quit(status = 1)
    })
}

main <- function() {
    cat("Starting spatial data processing...\n")
    process_spatial_data(
        cellinfo_path = opt$cellinfo,
        geneinfo_path = opt$geneinfo,
        matrix_path = opt$matrix,
        loc_path = opt$spatial_loc,
        output_dir = opt$output_dir
    )
}

main()

