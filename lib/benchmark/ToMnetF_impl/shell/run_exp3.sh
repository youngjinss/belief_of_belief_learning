#!/bin/bash
# nohup bash shell/run_exp3.sh all > experiment3.log 2>&1 &
# Complete workflow automation for ToMnetF experiment3 with N_past functionality
# Usage: bash run_exp3.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=3
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
DATA_DIR="$BASE_DIR/data/experiment3"
MODELS_DIR="$BASE_DIR/models/experiment3"
RESULTS_DIR="$BASE_DIR/result/experiment3"
PLOTS_DIR="$BASE_DIR/plots/experiment3"
LOG_DIR="$BASE_DIR/log"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$LOG_DIR/experiment3/$TIMESTAMP"

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"


# Parse command line arguments
COMMAND=${1:-all}

print_usage() {
    echo "Usage: $0 [data_generation|train|evaluate|visualize|n_past_eval|all|test]"
    echo ""
    echo "Commands:"
    echo "  data_generation  Generate trajectory data with SR, consumption labels, and N_past episodes"
    echo "  train           Train ToMnet model for experiment 3 with N_past character embedding"
    echo "  evaluate        Evaluate trained model"
    echo "  visualize       Create plots and visualizations including SR maps and N_past analysis"
    echo "  n_past_eval     Evaluate model performance across different N_past values (0-10)"
    echo "  all             Run complete pipeline"
    echo "  test            Run small test to verify modifications"
    echo ""
    echo "Experiment 3 Features:"
    echo "  - N_past episodes for character embedding (N_past ~ U{0, 10})"
    echo "  - Character embedding: e_char,i = sum(e_char,ij) for j=1..N_past"
    echo "  - Each past episode is a single state-action pair"
    echo "  - Enhanced logging: real-time output, component losses, epoch timing"
    echo "  - Enhanced visualization: component loss plots, N_past analysis"
    echo ""
}

log_step() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message"
    echo "$message" >> "$RUN_LOG_DIR/execution.log"
}

run_data_generation() {
    # Check if data already exists
    if [ -f "$DATA_DIR/processed_data_exp3.pkl" ]; then
        log_step "Data generation skipped - processed_data_exp3.pkl already exists"
        return 0
    fi
    
    log_step "Starting data generation for experiment $EXPERIMENT_NO with N_past functionality"
    log_step "Logging data generation output to: $RUN_LOG_DIR/data_generation.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python generate.py > "$RUN_LOG_DIR/data_generation.log" 2>&1
    
    log_step "Data generation completed"
    
    # Log N_past statistics if available
    if [ -f "$RUN_LOG_DIR/data_generation.log" ]; then
        log_step "N_past data generation summary:"
        grep -i "n_past\|past.*episode" "$RUN_LOG_DIR/data_generation.log" | tail -5 || true
    fi
}

run_training() {
    # Check if training already completed
    if [ -f "$MODELS_DIR/exp3_best.pth" ] && [ -f "$RESULTS_DIR/exp3_results.json" ]; then
        log_step "Training skipped - exp3_best.pth and exp3_results.json already exist"
        return 0
    fi
    
    log_step "Starting training for experiment $EXPERIMENT_NO with N_past character embedding"
    log_step "Logging training output to: $RUN_LOG_DIR/training.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python train.py --log_dir "$RUN_LOG_DIR/training.log" > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Training completed"
    
    # Log training summary if available
    if [ -f "$RUN_LOG_DIR/training.log" ]; then
        log_step "Training summary:"
        grep -i "epoch\|accuracy\|loss\|n_past" "$RUN_LOG_DIR/training.log" | tail -10 || true
    fi
}

run_evaluation() {
    # Check if evaluation already completed
    if [ -f "$RESULTS_DIR/cross_species_evaluation_exp3.json" ]; then
        log_step "Evaluation skipped - cross_species_evaluation_exp3.json already exists"
        return 0
    fi
    
    log_step "Starting evaluation for experiment $EXPERIMENT_NO"
    log_step "Logging evaluation output to: $RUN_LOG_DIR/evaluation.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python evaluate.py > "$RUN_LOG_DIR/evaluation.log" 2>&1
    
    log_step "Evaluation completed"
    
    # Log evaluation summary if available
    if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
        log_step "Evaluation summary:"
        grep -i "accuracy\|performance\|n_past" "$RUN_LOG_DIR/evaluation.log" | tail -5 || true
    fi
}

run_visualization() {
    # Check if visualization already completed
    if [ -f "$PLOTS_DIR/training_curves_exp3.png" ] && [ -f "$PLOTS_DIR/confusion_matrix_exp3.png" ]; then
        log_step "Visualization skipped - training_curves_exp3.png and confusion_matrix_exp3.png already exist"
        return 0
    fi
    
    log_step "Starting visualization for experiment $EXPERIMENT_NO"
    log_step "Logging visualization output to: $RUN_LOG_DIR/visualization.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python visualize.py > "$RUN_LOG_DIR/visualization.log" 2>&1
    
    log_step "Visualization completed"
}

run_n_past_evaluation() {
    # Check if N_past evaluation already completed
    if [ -f "$RESULTS_DIR/n_past_evaluation_results.json" ]; then
        log_step "N_past evaluation skipped - n_past_evaluation_results.json already exists"
        return 0
    fi
    
    log_step "Starting N_past evaluation for experiment $EXPERIMENT_NO"
    log_step "Evaluating model performance across different N_past values (0-10)"
    log_step "Logging N_past evaluation output to: $RUN_LOG_DIR/n_past_evaluation.log"
    
    cd "$SCRIPTS_DIR/experiment3"
    python evaluate.py --n_past_eval \
        --model_paths "$MODELS_DIR/exp3_best.pth" \
        --test_data_paths "$DATA_DIR/processed_data_exp3.pkl" \
        --n_past_min 0 --n_past_max 10 > "$RUN_LOG_DIR/n_past_evaluation.log" 2>&1
    
    log_step "N_past evaluation completed"
    
    # Log N_past evaluation summary if available
    if [ -f "$RUN_LOG_DIR/n_past_evaluation.log" ]; then
        log_step "N_past evaluation summary:"
        grep -i "accuracy\|n_past.*performance" "$RUN_LOG_DIR/n_past_evaluation.log" | tail -5 || true
    fi
}

# Additional function to check N_past implementation
check_n_past_implementation() {
    log_step "Checking N_past implementation..."
    
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
    
    log_step "N_past implementation check completed"
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
    n_past_eval)
        run_n_past_evaluation
        ;;
    all)
        log_step "Running complete ToMnetF pipeline for experiment $EXPERIMENT_NO with N_past functionality"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        check_n_past_implementation
        run_data_generation
        run_training
        run_evaluation
        run_visualization
        run_n_past_evaluation
        log_step "Complete pipeline finished successfully"
        ;;
    test)
        log_step "Running small test to verify modifications"
        bash shell/test_exp3_modifications.sh
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

log_step "Script completed successfully"
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
echo "Experiment 3 N_past Features:"
echo "  - Character embedding uses past episodes: e_char,i = sum(e_char,ij)"
echo "  - N_past sampled from U{0, 10} for each game"
echo "  - Each past episode is a single state-action pair"
echo "  - When N_past = 0, past episode contribution is zero"