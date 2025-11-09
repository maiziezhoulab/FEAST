#!/bin/bash

# --- Configuration ---
INPUT_DIR="/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure2_clustering/grad_76_sim_for_var"
OUTPUT_DIR_BASE="/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure2_clustering/results_multi_iterations_76_for_var"
N_CLUSTERS_PARAM=7
K_NEIGHBORS=6  # Number of neighbors for SEDR spatial graph

# --- Python Script Paths ---
PYTHON_SCRIPT_STAGATE="/maiziezhou_lab6/chen_yr/clustering/cluster_script/clustering_STAGATE_mclust.py"
PYTHON_SCRIPT_GRAPHST="/maiziezhou_lab6/chen_yr/clustering/cluster_script/clustering_graphST.py"
PYTHON_SCRIPT_SEDR="/maiziezhou_lab6/chen_yr/clustering/cluster_script/clustering_SEDR.py"
PYTHON_SCRIPT_CONSOLIDATE="/maiziezhou_lab6/chen_yr/clustering/cluster_script/consolidate_results.py"

# --- Conda Environment Paths ---
ENV_STAGATE="/maiziezhou_lab6/chen_yr/miniconda3/envs/STAGATE"
ENV_GRAPHST="/maiziezhou_lab6/chen_yr/miniconda3/envs/GraphST"
ENV_SEDR="/maiziezhou_lab6/chen_yr/miniconda3/envs/SEDR_env"
# The consolidation script uses pandas, which is in the STAGATE env.
ENV_CONSOLIDATE="$ENV_STAGATE"

# Create log directory if it doesn't exist
mkdir -p "logs"

# --- Main Sequential Execution Loop ---
# This loop will now run 6 independent jobs one after another.
for i in {1..6}
do
    ITERATION_NAME="run_${i}"
    LOG_FILE="logs/iterative_clustering_local_run_${i}.out"

    echo "====================================================="
    echo "Starting Clustering Pipeline for ITERATION: $ITERATION_NAME"
    echo "Log file: $LOG_FILE"
    echo "====================================================="

    # --- Main Processing Loop for this iteration ---
    # Redirect stdout and stderr for the entire processing block to a log file.
    {
        find "$INPUT_DIR" -type f -name "*.h5ad" | while read -r INPUT_H5AD; do
            RELATIVE_PATH="${INPUT_H5AD#$INPUT_DIR/}"
            RELATIVE_PATH_NO_EXT="${RELATIVE_PATH%.h5ad}"
            BASENAME=$(basename "$INPUT_H5AD" .h5ad)

            FINAL_OUTPUT_DIR="$OUTPUT_DIR_BASE/$RELATIVE_PATH_NO_EXT/$ITERATION_NAME"
            mkdir -p "$FINAL_OUTPUT_DIR"
            
            # Define paths for intermediate and final CSV files
            TMP_STAGATE_CSV="$FINAL_OUTPUT_DIR/${BASENAME}_${ITERATION_NAME}_stagate.csv"
            TMP_GRAPHST_CSV="$FINAL_OUTPUT_DIR/${BASENAME}_${ITERATION_NAME}_graphst.csv"
            TMP_SEDR_CSV="$FINAL_OUTPUT_DIR/${BASENAME}_${ITERATION_NAME}_sedr.csv"
            FINAL_CSV_FILE="$FINAL_OUTPUT_DIR/${BASENAME}_${ITERATION_NAME}_all_clusters.csv"
            
            echo "--- Processing Sample: $BASENAME (Iteration: $ITERATION_NAME) ---"
            echo "Input H5AD: $INPUT_H5AD"
            echo "Final Output CSV: $FINAL_CSV_FILE"

            # Clean up previous intermediate files if they exist
            rm -f "$TMP_STAGATE_CSV" "$TMP_GRAPHST_CSV" "$TMP_SEDR_CSV" "$FINAL_CSV_FILE"

            # --- Run STAGATE ---
            echo "  (1/3) Running STAGATE..."
            conda run -p "$ENV_STAGATE" python3 "$PYTHON_SCRIPT_STAGATE" \
                --input "$INPUT_H5AD" \
                --num_cluster "$N_CLUSTERS_PARAM" \
                --save_df_path "$TMP_STAGATE_CSV"
            if [ $? -ne 0 ]; then echo "  ERROR: STAGATE failed for $BASENAME on iteration $ITERATION_NAME"; continue; fi

            # --- Run GraphST ---
            echo "  (2/3) Running GraphST..."
            conda run -p "$ENV_GRAPHST" python3 "$PYTHON_SCRIPT_GRAPHST" \
                --input "$INPUT_H5AD" \
                --n_clusters "$N_CLUSTERS_PARAM" \
                --tool "mclust" \
                --save_df_path "$TMP_GRAPHST_CSV"
            if [ $? -ne 0 ]; then echo "  ERROR: GraphST failed for $BASENAME on iteration $ITERATION_NAME"; continue; fi


            # --- Consolidate Results ---
            echo "  (3/3) Consolidating results..."
            conda run -p "$ENV_CONSOLIDATE" python3 "$PYTHON_SCRIPT_CONSOLIDATE" \
                --output "$FINAL_CSV_FILE" \
                "$TMP_STAGATE_CSV" "$TMP_GRAPHST_CSV" "$TMP_SEDR_CSV"
            if [ $? -ne 0 ]; then echo "  ERROR: Consolidation failed for $BASENAME on iteration $ITERATION_NAME"; continue; fi

            # --- Clean up temporary files ---
            rm -f "$TMP_STAGATE_CSV" "$TMP_GRAPHST_CSV" "$TMP_SEDR_CSV"

            echo "  SUCCESS: Finished $BASENAME. Results are in $FINAL_CSV_FILE"
            echo "-----------------------------------------------------"
        done

        echo "====================================================="
        echo "All samples processed for ITERATION: $ITERATION_NAME"
        echo "====================================================="

    } > "$LOG_FILE" 2>&1

done

echo "All sequential runs have completed."