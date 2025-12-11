#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --config <config_path> --output_root <output_root> [--dataset <dataset_name>] [--n <num_top_dirs>] [--log_dir <log_dir>]"
    echo "  --config: Path to config.ini (required)"
    echo "  --output_root: Root directory for output (required)"
    echo "  --dataset: Dataset name (default: dataset1)"
    echo "  --n: Process first n top-level directories (for debugging, default: all)"
    echo "  --log_dir: Directory for logs (default: <output_root>/logs)"
    exit 1
}

# Parse arguments
CONFIG=""
OUTPUT_ROOT=""
DATASET="dataset1"
N=-1  # -1 means all
LOG_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        *) usage ;;
    esac
done

# Check required args
if [ -z "$CONFIG" ] || [ -z "$OUTPUT_ROOT" ]; then
    usage
fi

# Set default log dir
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$OUTPUT_ROOT/logs"
fi
mkdir -p "$LOG_DIR"

# Python script path (fixed as per example)
PYTHON_SCRIPT="/mnt/raid/luohaoran/cicids2018/SaP/phased_dataset_gen/src/pipline_cic_ids_2018_MP.py"

# Input root (fixed as current dir from tree)
INPUT_ROOT="."

# Get top-level dirs (length, loss, etc.), sorted alphabetically
TOP_DIRS=($(ls -d */ | sed 's#/##g' | sort))

# Limit to first n if specified
if [ $N -gt 0 ]; then
    TOP_DIRS=("${TOP_DIRS[@]:0:$N}")
fi

# Counter for background jobs to wait periodically
JOB_COUNT=0
MAX_JOBS=3  # Limit parallel jobs to avoid overload, adjust as needed

# Record start time
START_TIME=$(date +%s)

for top_dir in "${TOP_DIRS[@]}"; do
    # Get sub-dirs under top_dir, exclude *-flow and logs
    SUB_DIRS=($(ls -d "$INPUT_ROOT/$top_dir"/*/ | sed 's#.*/\([^/]*\)/#\1#g' | grep -v -E '(-flow|logs)$' | sort))
    
    for sub_dir in "${SUB_DIRS[@]}"; do
        # Get date dirs under sub_dir, exclude *-flow and logs
        DATE_DIRS=($(ls -d "$INPUT_ROOT/$top_dir/$sub_dir"/*/ | sed 's#.*/\([^/]*\)/#\1#g' | grep -v -E '(-flow|logs)$' | sort))
        
        for date_dir in "${DATE_DIRS[@]}"; do
            # Build paths
            INPUT_DIR="$INPUT_ROOT/$top_dir/$sub_dir/$date_dir"
            OUTPUT_DIR="$OUTPUT_ROOT/$top_dir/$sub_dir/$date_dir"
            mkdir -p "$OUTPUT_DIR"
            
            # Generate unique log file with timestamp
            TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
            LOG_FILE="$LOG_DIR/${top_dir}_${sub_dir}_${date_dir}_${TIMESTAMP}.txt"
            
            # Run the command in background, redirect stdout/stderr to log
            python "$PYTHON_SCRIPT" --config "$CONFIG" --input_dir "$INPUT_DIR" --dataset "$DATASET" --output_dir "$OUTPUT_DIR" --run > "$LOG_FILE" 2>&1 &
            
            echo "Started processing: $top_dir/$sub_dir/$date_dir, log at $LOG_FILE"
            
            # Increment job count and wait if reached max
            JOB_COUNT=$((JOB_COUNT + 1))
            if [ $JOB_COUNT -ge $MAX_JOBS ]; then
                wait  # Wait for all background jobs to finish
                JOB_COUNT=0
            fi
        done
    done
done

# Wait for any remaining jobs
wait

echo "All processing completed. Logs are in $LOG_DIR"

# Record end time and calculate total execution time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "Total execution time: $TOTAL_TIME seconds"