#!/bin/bash
# nohup bash shell/run_exp3_small.sh all > experiment3_small.log 2>&1 &
# Small-scale workflow automation for ToMnetF experiment3 with reduced parameters
# Usage: bash run_exp3_small.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=3
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
DATA_DIR="$BASE_DIR/data/experiment3_small"
MODELS_DIR="$BASE_DIR/models/experiment3_small"
RESULTS_DIR="$BASE_DIR/result/experiment3_small"
PLOTS_DIR="$BASE_DIR/plots/experiment3_small"
LOG_DIR="$BASE_DIR/log"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$LOG_DIR/experiment3_small/$TIMESTAMP"

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"

# Parse command line arguments
COMMAND=${1:-all}

print_usage() {
    echo "Usage: $0 [data_generation|train|evaluate|visualize|all]"
    echo ""
    echo "Commands:"
    echo "  data_generation  Generate small trajectory data with SR, consumption labels, and N_past episodes"
    echo "  train           Train ToMnet model for experiment 3 with reduced epochs and batch size"
    echo "  evaluate        Evaluate trained model"
    echo "  visualize       Create plots and visualizations including SR maps and N_past analysis"
    echo "  all             Run complete pipeline"
    echo ""
    echo "Small Experiment Features:"
    echo "  - Reduced training epochs (10-20 instead of 100+)"
    echo "  - Smaller dataset size"
    echo "  - Faster execution for testing purposes"
    echo "  - N_past episodes for character embedding (N_past ~ U{0, 5})"
    echo ""
}

log_step() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message"
    echo "$message" >> "$RUN_LOG_DIR/execution.log"
}

run_data_generation() {
    # Check if data already exists
    if [ -f "$DATA_DIR/processed_data_exp3_small.pkl" ]; then
        log_step "Data generation skipped - processed_data_exp3_small.pkl already exists"
        return 0
    fi
    
    log_step "Starting small data generation for experiment $EXPERIMENT_NO with N_past functionality"
    log_step "Logging data generation output to: $RUN_LOG_DIR/data_generation.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    # Run with small dataset parameters
    python data_generation.py --small_experiment --output_dir "$DATA_DIR" > "$RUN_LOG_DIR/data_generation.log" 2>&1
    
    log_step "Small data generation completed"
    
    # Log N_past statistics if available
    if [ -f "$RUN_LOG_DIR/data_generation.log" ]; then
        log_step "N_past data generation summary:"
        grep -i "n_past\|past.*episode\|total.*generated" "$RUN_LOG_DIR/data_generation.log" | tail -5 || true
    fi
}

run_training() {
    # Check if training already completed
    if [ -f "$MODELS_DIR/exp3_small_best.pth" ] && [ -f "$RESULTS_DIR/exp3_small_results.json" ]; then
        log_step "Training skipped - exp3_small_best.pth and exp3_small_results.json already exist"
        return 0
    fi
    
    log_step "Starting small training for experiment $EXPERIMENT_NO with N_past character embedding"
    log_step "Logging training output to: $RUN_LOG_DIR/training.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    # Run with small training parameters
    python train.py --small_experiment --data_dir "$DATA_DIR" --model_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR" > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Small training completed"
    
    # Log training summary if available
    if [ -f "$RUN_LOG_DIR/training.log" ]; then
        log_step "Training summary:"
        grep -E "Epoch:.*Train Loss:.*Train Acc:.*Val Acc:" "$RUN_LOG_DIR/training.log" | tail -5 || true
        grep -i "finished.*training\|best.*accuracy" "$RUN_LOG_DIR/training.log" | tail -3 || true
    fi
}

run_evaluation() {
    # Check if evaluation already completed
    if [ -f "$RESULTS_DIR/cross_species_evaluation_exp3_small.json" ]; then
        log_step "Evaluation skipped - cross_species_evaluation_exp3_small.json already exists"
        return 0
    fi
    
    log_step "Starting evaluation for small experiment $EXPERIMENT_NO"
    log_step "Logging evaluation output to: $RUN_LOG_DIR/evaluation.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python evaluate.py --small_experiment --model_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR" > "$RUN_LOG_DIR/evaluation.log" 2>&1
    
    log_step "Small evaluation completed"
    
    # Log evaluation summary if available
    if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
        log_step "Evaluation summary:"
        grep -i "accuracy\|performance\|n_past" "$RUN_LOG_DIR/evaluation.log" | tail -5 || true
    fi
}

run_visualization() {
    # Check if visualization already completed
    if [ -f "$PLOTS_DIR/training_curves_exp3_small.png" ] && [ -f "$PLOTS_DIR/confusion_matrix_exp3_small.png" ]; then
        log_step "Visualization skipped - training_curves_exp3_small.png and confusion_matrix_exp3_small.png already exist"
        return 0
    fi
    
    log_step "Starting visualization for small experiment $EXPERIMENT_NO"
    log_step "Logging visualization output to: $RUN_LOG_DIR/visualization.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python visualize.py --small_experiment --results_dir "$RESULTS_DIR" --plots_dir "$PLOTS_DIR" > "$RUN_LOG_DIR/visualization.log" 2>&1
    
    log_step "Small visualization completed"
}

# Additional function to check N_past implementation
check_n_past_implementation() {
    log_step "Checking N_past implementation for small experiment..."
    
    cd "$SCRIPTS_DIR/experiment3"
    
    # Check if config has N_past parameters
    if grep -q "n_past" config.py; then
        log_step "✓ N_past parameters found in config.py"
    else
        log_step "✗ N_past parameters not found in config.py"
    fi
    
    # Check if tomnet.py has N_past support
    if grep -q "past_episodes" tomnet.py; then
        log_step "✓ N_past support found in tomnet.py"
    else
        log_step "✗ N_past support not found in tomnet.py"
    fi
    
    # Check if data_generation.py exists
    if [ -f "data_generation.py" ]; then
        log_step "✓ data_generation.py exists"
    else
        log_step "✗ data_generation.py not found"
    fi
    
    log_step "N_past implementation check completed for small experiment"
}

# Main execution
case $COMMAND in
    data_generation)
        check_n_past_implementation
        run_data_generation
        ;;
    train)
        run_training
        ;;
    evaluate)
        run_evaluation
        ;;
    visualize)
        run_visualization
        ;;
    all)
        log_step "Running complete ToMnetF small pipeline for experiment $EXPERIMENT_NO with N_past functionality"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        check_n_past_implementation
        run_data_generation
        run_training
        run_evaluation
        run_visualization
        log_step "Complete small pipeline finished successfully"
        ;;
    check)
        check_n_past_implementation
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        print_usage
        exit 1
        ;;
esac

log_step "Small experiment script completed successfully"
log_step "Log files saved to: $RUN_LOG_DIR/"
echo ""
echo "Log files created:"
echo "  - $RUN_LOG_DIR/execution.log (main script execution log)"
if [ -f "$RUN_LOG_DIR/data_generation.log" ]; then
    echo "  - $RUN_LOG_DIR/data_generation.log"
fi
if [ -f "$RUN_LOG_DIR/training.log" ]; then
    echo "  - $RUN_LOG_DIR/training.log"
fi
if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
    echo "  - $RUN_LOG_DIR/evaluation.log"
fi
if [ -f "$RUN_LOG_DIR/visualization.log" ]; then
    echo "  - $RUN_LOG_DIR/visualization.log"
fi

echo ""
echo "Small Experiment 3 Features:"
echo "  - Reduced training epochs for faster execution"
echo "  - Smaller dataset size"
echo "  - Character embedding uses past episodes: e_char,i = sum(e_char,ij)"
echo "  - N_past sampled from U{0, 5} for each game (reduced from U{0, 10})"
echo "  - Each past episode is a single state-action pair"
echo "  - When N_past = 0, past episode contribution is zero"