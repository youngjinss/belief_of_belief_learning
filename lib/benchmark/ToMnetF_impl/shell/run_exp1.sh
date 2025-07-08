#!/bin/bash
# nohup bash shell/run_exp1.sh all > experiment.log 2>&1 &
# Complete workflow automation for ToMnetF experiment1
# Usage: bash run_exp1.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=1
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
DATA_DIR="$BASE_DIR/data/experiment1"
MODELS_DIR="$BASE_DIR/models/experiment1"
RESULTS_DIR="$BASE_DIR/result/experiment1"
PLOTS_DIR="$BASE_DIR/plots/experiment1"
LOG_DIR="$BASE_DIR/log"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$LOG_DIR/experiment1/$TIMESTAMP"

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"

# Parse command line arguments
COMMAND=${1:-all}

print_usage() {
    echo "Usage: $0 [data_generation|train|evaluate|visualize|all]"
    echo ""
    echo "Commands:"
    echo "  data_generation  Generate trajectory data"
    echo "  train           Train ToMnet model"
    echo "  evaluate        Evaluate trained model"
    echo "  visualize       Create plots and visualizations"
    echo "  all             Run complete pipeline"
}

log_step() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message"
    echo "$message" >> "$RUN_LOG_DIR/execution.log"
}

run_data_generation() {
    # Check if data already exists
    if [ -f "$DATA_DIR/processed_data_exp1.pkl" ]; then
        log_step "Data generation skipped - processed_data_exp1.pkl already exists"
        return 0
    fi
    
    log_step "Starting data generation for experiment $EXPERIMENT_NO"
    log_step "Logging data generation output to: $RUN_LOG_DIR/data_generation.log"
    
    cd "$SCRIPTS_DIR/experiment1"
    python generate.py > "$RUN_LOG_DIR/data_generation.log" 2>&1
    
    log_step "Data generation completed"
}

run_training() {
    # Check if training already completed
    if [ -f "$MODELS_DIR/exp1_best.pth" ] && [ -f "$RESULTS_DIR/exp1_results.json" ]; then
        log_step "Training skipped - exp1_best.pth and exp1_results.json already exist"
        return 0
    fi
    
    log_step "Starting training for experiment $EXPERIMENT_NO"
    log_step "Logging training output to: $RUN_LOG_DIR/training.log"
    
    cd "$SCRIPTS_DIR/experiment1"
    python train.py > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Training completed"
}

run_evaluation() {
    # Check if evaluation already completed
    if [ -f "$RESULTS_DIR/cross_species_evaluation_exp1.json" ]; then
        log_step "Evaluation skipped - cross_species_evaluation_exp1.json already exists"
        return 0
    fi
    
    log_step "Starting evaluation for experiment $EXPERIMENT_NO"
    log_step "Logging evaluation output to: $RUN_LOG_DIR/evaluation.log"
    
    cd "$SCRIPTS_DIR/experiment1"
    python evaluate.py > "$RUN_LOG_DIR/evaluation.log" 2>&1
    
    log_step "Evaluation completed"
}

run_visualization() {
    # Check if visualization already completed
    if [ -f "$PLOTS_DIR/training_curves_exp1.png" ] && [ -f "$PLOTS_DIR/confusion_matrix_exp1.png" ]; then
        log_step "Visualization skipped - training_curves_exp1.png and confusion_matrix_exp1.png already exist"
        return 0
    fi
    
    log_step "Starting visualization for experiment $EXPERIMENT_NO"
    log_step "Logging visualization output to: $RUN_LOG_DIR/visualization.log"
    
    cd "$SCRIPTS_DIR/experiment1"
    python visualize.py \
        --experiment_no "$EXPERIMENT_NO" \
        --result_dir "../../result/experiment1" \
        --plot_dir "../../plots/experiment1" \
        --plot_type "all" \
        > "$RUN_LOG_DIR/visualization.log" 2>&1
    
    log_step "Visualization completed"
}

# Main execution
case $COMMAND in
    data_generation)
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
        log_step "Running complete ToMnetF pipeline for experiment $EXPERIMENT_NO"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        run_data_generation
        run_training
        run_evaluation
        run_visualization
        log_step "Complete pipeline finished successfully"
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