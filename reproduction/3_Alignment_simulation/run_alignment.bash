# Print debugging information
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================="

# Define base paths
DATA_DIR="/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure3_alignment/grad_alignment_sim"
SCRIPTS_DIR="/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure3_alignment/alignment_scripts"
OUTPUT_BASE_DIR="/maiziezhou_lab6/chen_yr/scripts/2Reproduce/Figure3_alignment/alignment_results_change"

# Function to check and create directory if it doesn't exist
ensure_dir_exists() {
    local dir_path=$1
    if [ ! -d "$dir_path" ]; then
        echo "Creating directory: $dir_path"
        mkdir -p "$dir_path"
    fi
}

# Create output base directory
ensure_dir_exists "$OUTPUT_BASE_DIR"

# Available angles
ANGLES=( "1" "5" "10" "30" "45" "60" )

# Initialize counters for reporting
TOTAL_JOBS=0
SUCCESSFUL_JOBS=0
FAILED_JOBS=0
SKIPPED_JOBS=0

# Create summary report file
REPORT_FILE="${OUTPUT_BASE_DIR}/alignment_summary_report.txt"
ensure_dir_exists "$(dirname "$REPORT_FILE")"
echo "Alignment Summary Report - $(date)" > "$REPORT_FILE"
echo "======================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Function to run a single alignment tool on a single angle
run_alignment() {
    local algorithm=$1
    local env_name=$2
    local script_name=$3
    local reference=$4
    local transformed=$5
    local output_dir=$6
    
    # Create output directory for this algorithm
    ensure_dir_exists "$output_dir/$algorithm"
    
    # Check if environment exists
    local env_path=$(echo "$env_name" | sed 's/-p //')
    if [ ! -d "$env_path" ]; then
        echo "WARNING: Conda environment $env_path not found. Skipping ${algorithm}."
        echo "$(date): SKIPPED - ${algorithm} - Environment not found: $env_path" >> "$REPORT_FILE"
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
        return 1
    fi
    
    # Check if script exists
    if [ ! -f "${SCRIPTS_DIR}/${script_name}" ]; then
        echo "WARNING: Script ${SCRIPTS_DIR}/${script_name} not found. Skipping ${algorithm}."
        echo "$(date): SKIPPED - ${algorithm} - Script not found" >> "$REPORT_FILE"
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
        return 1
    fi
    
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    
    echo "Running ${algorithm} alignment..."
    
    # Temporarily disable strict error checking for alignment commands
    set +e
    
    # Add specific arguments for different algorithms
    extra_args=""
    if [ "$algorithm" = "Spacel" ]; then
        extra_args="--layer_key sce.layer_guess"
    fi
    
    conda run ${env_name} python ${SCRIPTS_DIR}/${script_name} \
        --slice1 "${reference}" \
        --slice2 "${transformed}" \
        --output_dir "${output_dir}/${algorithm}" \
        ${extra_args}
    local exit_code=$?
    set -e
    
    if [ $exit_code -eq 0 ]; then
        echo "${algorithm} alignment completed successfully"
        echo "$(date): SUCCESS - ${algorithm} - $(basename ${reference}) vs $(basename ${transformed})" >> "$REPORT_FILE"
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
        return 0
    else
        echo "Error running ${algorithm} alignment on $(basename ${transformed}) (exit code: $exit_code)"
        echo "$(date): FAILED - ${algorithm} - $(basename ${reference}) vs $(basename ${transformed}) (exit code: $exit_code)" >> "$REPORT_FILE"
        FAILED_JOBS=$((FAILED_JOBS + 1))
        return 1
    fi
}

# Process each angle
for angle in "${ANGLES[@]}"; do
    echo "=== Processing angle: ${angle} degrees ==="
    
    # Define file paths
    REFERENCE_FILE="${DATA_DIR}/original_adata.h5ad"
    TRANSFORMED_FILE="${DATA_DIR}/transformed_adata_angle_${angle}.h5ad"
    
    # Check if files exist
    if [ ! -f "$REFERENCE_FILE" ]; then
        echo "WARNING: Reference file not found: $REFERENCE_FILE"
        echo "$(date): SKIPPED - angle_${angle} - Reference file not found" >> "$REPORT_FILE"
        continue
    fi
    
    if [ ! -f "$TRANSFORMED_FILE" ]; then
        echo "WARNING: Transformed file not found: $TRANSFORMED_FILE"
        echo "$(date): SKIPPED - angle_${angle} - Transformed file not found" >> "$REPORT_FILE"
        continue
    fi
    
    # Create output directory for this angle
    ANGLE_OUTPUT_DIR="${OUTPUT_BASE_DIR}/angle_${angle}"
    ensure_dir_exists "$ANGLE_OUTPUT_DIR"
    
    # Copy the reference and transformed slices to the output directory
    echo "  Copying files to output directory..."
    cp "$REFERENCE_FILE" "${ANGLE_OUTPUT_DIR}/reference.h5ad"
    cp "$TRANSFORMED_FILE" "${ANGLE_OUTPUT_DIR}/transformed.h5ad"
    
    # Disable strict error checking for alignment runs
    set +e
    
    # Run alignments
    run_alignment "Spacel" "-p /maiziezhou_lab6/chen_yr/miniconda3/envs/SPACEL" "run_spacel.py" "$REFERENCE_FILE" "$TRANSFORMED_FILE" "$ANGLE_OUTPUT_DIR"
    run_alignment "Spateo" "-p /maiziezhou_lab6/chen_yr/miniconda3/envs/simulator" "run_spateo.py" "$REFERENCE_FILE" "$TRANSFORMED_FILE" "$ANGLE_OUTPUT_DIR"
    
    # Re-enable strict error checking
    set -e
    
    echo "=== Completed angle: ${angle} degrees ==="
done

# Write final summary to report
echo "" >> "$REPORT_FILE"
echo "======================================" >> "$REPORT_FILE"
echo "FINAL SUMMARY:" >> "$REPORT_FILE"
echo "Total Jobs: $TOTAL_JOBS" >> "$REPORT_FILE"
echo "Successful: $SUCCESSFUL_JOBS" >> "$REPORT_FILE"
echo "Failed: $FAILED_JOBS" >> "$REPORT_FILE"
echo "Skipped: $SKIPPED_JOBS" >> "$REPORT_FILE"
echo "Success Rate: $(( SUCCESSFUL_JOBS * 100 / (TOTAL_JOBS > 0 ? TOTAL_JOBS : 1) ))%" >> "$REPORT_FILE"
echo "======================================" >> "$REPORT_FILE"

echo "All angles processed!"
echo "Summary: Total: $TOTAL_JOBS, Success: $SUCCESSFUL_JOBS, Failed: $FAILED_JOBS, Skipped: $SKIPPED_JOBS"
