#!/bin/bash
# Parallel ToMnet experiment launcher
# nohup bash shell/run_exp3_parallel.sh all > experiment.log 2>&1 &
# This version trains multiple alpha values in parallel to reduce execution time

set -e  # Exit on error

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
ToMnet Experiment Launcher (Parallel Version)

DESCRIPTION:
    Complete workflow for running ToMnet experiments with parallel training.
    This version trains multiple alpha values simultaneously to reduce execution time.

USAGE:
    bash run_exp3_parallel.sh [COMMAND] [OPTIONS]

COMMANDS:
    train [OPTIONS]         Train ToMnet models in parallel for different alpha values
    evaluate [OPTIONS]      Run cross-species evaluation
    visualize [OPTIONS]     Generate Figure 3 visualizations
    all [OPTIONS]           Run complete workflow (parallel train + evaluate + visualize)
    clean                   Clean up generated files
    help                    Show this help message

PARALLEL TRAINING:
    - Automatically detects available GPUs and CPU cores
    - Trains different alpha values simultaneously
    - Monitors progress and collects results
    - Significantly reduces total training time

EXAMPLES:
    # Complete workflow with parallel training
    bash run_exp3_parallel.sh all
    
    # Parallel training only
    bash run_exp3_parallel.sh train --n_epochs 30 --alpha_values 0.01 0.1 0.5
    
    # Force specific number of parallel jobs
    bash run_exp3_parallel.sh train --max_parallel 3

EOF
}

# Function to detect available resources
detect_resources() {
    # Detect GPUs
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    else
        GPU_COUNT=0
    fi
    
    # Detect CPU cores
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        # Linux
        CPU_CORES=$(nproc)
    fi
    
    # Calculate optimal parallel jobs
    if [ $GPU_COUNT -gt 0 ]; then
        # Use GPUs if available, but don't exceed 4 parallel jobs to avoid memory issues
        MAX_PARALLEL=$(( GPU_COUNT < 4 ? GPU_COUNT : 4 ))
        DEVICE_TYPE="GPU"
    else
        # Use CPU cores, but limit to 4 to avoid overwhelming the system
        MAX_PARALLEL=$(( CPU_CORES < 4 ? CPU_CORES : 4 ))
        DEVICE_TYPE="CPU"
    fi
    
    print_info "Resource detection:"
    print_info "  - Available GPUs: $GPU_COUNT"
    print_info "  - Available CPU cores: $CPU_CORES"
    print_info "  - Using device type: $DEVICE_TYPE"
    print_info "  - Max parallel jobs: $MAX_PARALLEL"
}

# Function to train a single alpha value (used in parallel)
train_single_alpha() {
    local alpha=$1
    local job_id=$2
    local total_jobs=$3
    local device_id=$4
    local log_dir=$5
    shift 5  # Remove the first 5 arguments, rest are training args
    
    local alpha_log_dir="$log_dir/alpha_${alpha}"
    mkdir -p "$alpha_log_dir"
    
    local alpha_log_file="$alpha_log_dir/training.log"
    local alpha_pid_file="$alpha_log_dir/process.pid"
    
    # Set device
    if [ $GPU_COUNT -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$device_id
        device_arg="--device cuda"
    else
        device_arg="--device cpu"
    fi
    
    echo "[$job_id/$total_jobs] Starting training for alpha=$alpha on device $device_id at $(date)" > "$alpha_log_file"
    echo "Arguments: $*" >> "$alpha_log_file"
    echo "" >> "$alpha_log_file"
    
    # Train single alpha value
    python scripts/train.py \
        --experiment figure3 \
        --alpha_values "$alpha" \
        $device_arg \
        "$@" >> "$alpha_log_file" 2>&1 &
    
    local train_pid=$!
    echo $train_pid > "$alpha_pid_file"
    
    # Wait for completion and return status
    wait $train_pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$total_jobs] ✓ Alpha=$alpha completed successfully at $(date)" >> "$alpha_log_file"
        echo "SUCCESS:$alpha" > "$alpha_log_dir/status"
    else
        echo "[$job_id/$total_jobs] ✗ Alpha=$alpha failed at $(date)" >> "$alpha_log_file"
        echo "FAILED:$alpha" > "$alpha_log_dir/status"
    fi
    
    return $exit_code
}

# Function to run parallel training
run_parallel_training() {
    print_info "Starting ToMnet parallel training..."
    
    # Parse alpha values from arguments
    local alpha_values=()
    local other_args=()
    local max_parallel_override=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --alpha_values)
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    alpha_values+=("$1")
                    shift
                done
                ;;
            --max_parallel)
                max_parallel_override="$2"
                shift 2
                ;;
            *)
                other_args+=("$1")
                shift
                ;;
        esac
    done
    
    # Use default alpha values if none specified
    if [ ${#alpha_values[@]} -eq 0 ]; then
        alpha_values=(0.01 0.03 0.1 0.3 1.0 3.0)
        print_info "Using default alpha values: ${alpha_values[*]}"
    else
        print_info "Using specified alpha values: ${alpha_values[*]}"
    fi
    
    # Detect resources
    detect_resources
    
    # Override max parallel if specified
    if [ -n "$max_parallel_override" ]; then
        MAX_PARALLEL=$max_parallel_override
        print_info "Overriding max parallel jobs to: $MAX_PARALLEL"
    fi
    
    # Create log directory with timestamp
    LOG_DIR="log/parallel_training/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Create main log file
    MAIN_LOG="$LOG_DIR/parallel_training.log"
    {
        echo "=== Parallel ToMnet Training Started at $(date) ==="
        echo "Alpha values: ${alpha_values[*]}"
        echo "Max parallel jobs: $MAX_PARALLEL"
        echo "Device type: $DEVICE_TYPE"
        echo "Additional arguments: ${other_args[*]}"
        echo ""
    } > "$MAIN_LOG"
    
    print_info "Starting parallel training for ${#alpha_values[@]} alpha values with up to $MAX_PARALLEL concurrent jobs"
    
    # Start training jobs in parallel
    local running_jobs=()
    local job_id=0
    local total_jobs=${#alpha_values[@]}
    
    for alpha in "${alpha_values[@]}"; do
        # Wait if we've reached the parallel limit
        while [ ${#running_jobs[@]} -ge $MAX_PARALLEL ]; do
            # Check completed jobs
            local new_running_jobs=()
            for job_info in "${running_jobs[@]}"; do
                local job_pid=$(echo "$job_info" | cut -d: -f1)
                local job_alpha=$(echo "$job_info" | cut -d: -f2)
                
                if kill -0 $job_pid 2>/dev/null; then
                    # Job still running
                    new_running_jobs+=("$job_info")
                else
                    # Job completed
                    wait $job_pid
                    local exit_code=$?
                    
                    if [ $exit_code -eq 0 ]; then
                        print_success "Alpha=$job_alpha training completed"
                    else
                        print_error "Alpha=$job_alpha training failed"
                    fi
                fi
            done
            running_jobs=("${new_running_jobs[@]}")
            
            if [ ${#running_jobs[@]} -ge $MAX_PARALLEL ]; then
                sleep 5  # Wait before checking again
            fi
        done
        
        # Start new job
        job_id=$((job_id + 1))
        device_id=$(( (job_id - 1) % MAX_PARALLEL ))
        
        print_info "[$job_id/$total_jobs] Starting training for alpha=$alpha on device $device_id"
        
        # Start training in background
        train_single_alpha "$alpha" "$job_id" "$total_jobs" "$device_id" "$LOG_DIR" "${other_args[@]}" &
        local train_pid=$!
        
        running_jobs+=("$train_pid:$alpha")
    done
    
    # Wait for all remaining jobs to complete
    print_info "Waiting for all training jobs to complete..."
    local failed_alphas=()
    
    for job_info in "${running_jobs[@]}"; do
        local job_pid=$(echo "$job_info" | cut -d: -f1)
        local job_alpha=$(echo "$job_info" | cut -d: -f2)
        
        wait $job_pid
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            print_success "Alpha=$job_alpha training completed"
        else
            print_error "Alpha=$job_alpha training failed"
            failed_alphas+=("$job_alpha")
        fi
    done
    
    # Check overall results
    echo "" >> "$MAIN_LOG"
    echo "=== Parallel Training Completed at $(date) ===" >> "$MAIN_LOG"
    
    if [ ${#failed_alphas[@]} -eq 0 ]; then
        print_success "All parallel training jobs completed successfully!"
        echo "All training jobs completed successfully" >> "$MAIN_LOG"
        
        # Generate unified cross-species evaluation files
        print_info "Generating unified cross-species evaluation files..."
        
        # Create unified model paths
        MODEL_PATHS_JSON="result/figure3/model_paths.json"
        mkdir -p "result/figure3"
        
        echo "{" > "$MODEL_PATHS_JSON"
        local first=true
        for alpha in "${alpha_values[@]}"; do
            if [ "$first" = false ]; then echo "," >> "$MODEL_PATHS_JSON"; fi
            echo "  \"alpha_$alpha\": \"models/figure3_${alpha}_best.pth\"" >> "$MODEL_PATHS_JSON"
            first=false
        done
        echo "}" >> "$MODEL_PATHS_JSON"
        
        # Create unified data paths
        DATA_PATHS_JSON="result/figure3/data_paths.json"
        echo "{" > "$DATA_PATHS_JSON"
        first=true
        for alpha in "${alpha_values[@]}"; do
            if [ "$first" = false ]; then echo "," >> "$DATA_PATHS_JSON"; fi
            echo "  \"alpha_$alpha\": \"data/figure3_alpha_${alpha}.pkl\"" >> "$DATA_PATHS_JSON"
            first=false
        done
        echo "}" >> "$DATA_PATHS_JSON"
        
        print_success "Cross-species evaluation files generated"
        
    else
        print_error "Some training jobs failed: ${failed_alphas[*]}"
        echo "Failed training jobs: ${failed_alphas[*]}" >> "$MAIN_LOG"
        
        print_info "Check individual log files in: $LOG_DIR"
        exit 1
    fi
    
    echo "" >> "$MAIN_LOG"
    echo "Log directory: $LOG_DIR" >> "$MAIN_LOG"
    echo "Individual alpha logs:" >> "$MAIN_LOG"
    for alpha in "${alpha_values[@]}"; do
        echo "  Alpha $alpha: $LOG_DIR/alpha_${alpha}/training.log" >> "$MAIN_LOG"
    done
    
    print_info "Parallel training completed! Check logs in: $LOG_DIR"
}

# Function to run evaluation (same as original)
run_evaluation() {
    print_info "Starting cross-species evaluation..."
    
    if [ -f "result/figure3/run_cross_species_evaluation.sh" ]; then
        # Create log directory with timestamp
        LOG_DIR="log/evaluation/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$LOG_DIR"
        print_info "Created log directory: $LOG_DIR"
        
        # Run evaluation in background with logging
        echo "Starting evaluation at $(date)" > "$LOG_DIR/execution.log"
        bash result/figure3/run_cross_species_evaluation.sh >> "$LOG_DIR/execution.log" 2>&1 &
        EVAL_PID=$!
        echo $EVAL_PID > "$LOG_DIR/process.pid"
        print_info "Evaluation started in background with PID: $EVAL_PID"
        print_info "Logs will be written to: $LOG_DIR/execution.log"
        
        # Wait for evaluation to complete
        wait $EVAL_PID
        if [ $? -eq 0 ]; then
            print_success "Evaluation completed!"
        else
            print_error "Evaluation failed! Check log: $LOG_DIR/execution.log"
            exit 1
        fi
    else
        print_error "Evaluation script not found. Run training first."
        exit 1
    fi
}

# Function to run visualization (same as original)
run_visualization() {
    print_info "Starting Figure 3 visualization..."
    
    # Create log directory with timestamp
    LOG_DIR="log/visualization/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Run visualization in background with logging
    echo "Starting visualization at $(date)" > "$LOG_DIR/execution.log"
    bash shell/visualize_figure3.sh "$@" >> "$LOG_DIR/execution.log" 2>&1 &
    VIS_PID=$!
    echo $VIS_PID > "$LOG_DIR/process.pid"
    print_info "Visualization started in background with PID: $VIS_PID"
    print_info "Logs will be written to: $LOG_DIR/execution.log"
    
    # Wait for visualization to complete
    wait $VIS_PID
    if [ $? -eq 0 ]; then
        print_success "Visualization completed!"
    else
        print_error "Visualization failed! Check log: $LOG_DIR/execution.log"
        exit 1
    fi
}

# Function to clean up generated files
clean_files() {
    print_warning "This will delete all generated data, models, and results."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up generated files..."
        rm -rf data/ models/ result/ plots/ log/
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift  # Remove first argument

case "$COMMAND" in
    train)
        run_parallel_training "$@"
        ;;
    evaluate)
        run_evaluation "$@"
        ;;
    visualize)
        run_visualization "$@"
        ;;
    all)
        print_info "Running complete ToMnet workflow with parallel training..."
        echo ""
        run_parallel_training "$@"
        echo ""
        run_evaluation "$@"
        echo ""
        run_visualization --save
        echo ""
        print_success "Complete parallel workflow finished!"
        ;;
    clean)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo "Use 'bash run_exp3_parallel.sh help' for usage information"
        exit 1
        ;;
esac