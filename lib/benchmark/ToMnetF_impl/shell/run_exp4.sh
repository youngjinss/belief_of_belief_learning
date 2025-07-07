#!/bin/bash
# nohup bash shell/run_exp4.sh all > experiment4.log 2>&1 &
# Complete workflow automation for ToMnetF experiment4 with random positions and goal rewards
# Usage: bash run_exp4.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=4
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
DATA_DIR="$BASE_DIR/data/experiment4"
MODELS_DIR="$BASE_DIR/models/experiment4"
RESULTS_DIR="$BASE_DIR/result/experiment4"
PLOTS_DIR="$BASE_DIR/plots/experiment4"
LOG_DIR="$BASE_DIR/log"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$LOG_DIR/experiment4/$TIMESTAMP"

# Create directories
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"


# Parse command line arguments
COMMAND=${1:-all}

print_usage() {
    echo "Usage: $0 [data_generation|train|evaluate|visualize|all]"
    echo ""
    echo "Commands:"
    echo "  data_generation  Generate trajectory data with random positions and goal rewards"
    echo "  train           Train ToMnet model for experiment 4"
    echo "  evaluate        Evaluate trained model"
    echo "  visualize       Create plots and visualizations including position and reward analysis"
    echo "  all             Run complete pipeline"
    echo ""
    echo "Experiment 4 Features:"
    echo "  - Random player positions (random_positions = True)"
    echo "  - Random goal rewards (random_goal_rewards = True)"
    echo "  - Extended environment variability for robust training"
    echo ""
}

log_step() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message"
    echo "$message" >> "$RUN_LOG_DIR/execution.log"
}

run_data_generation() {
    # Check if data already exists
    if [ -f "$DATA_DIR/processed_data_exp4.pkl" ]; then
        log_step "Data generation skipped - processed_data_exp4.pkl already exists"
        return 0
    fi
    
    log_step "Starting data generation for experiment $EXPERIMENT_NO with random positions and goal rewards"
    log_step "Logging data generation output to: $RUN_LOG_DIR/data_generation.log"
    
    cd "$SCRIPTS_DIR/experiment4"
    python generate.py > "$RUN_LOG_DIR/data_generation.log" 2>&1
    
    log_step "Data generation completed"
    
    # Log random positioning/reward statistics if available
    if [ -f "$RUN_LOG_DIR/data_generation.log" ]; then
        log_step "Random positioning and reward generation summary:"
        grep -i "random\|position\|reward\|goal" "$RUN_LOG_DIR/data_generation.log" | tail -5 || true
    fi
}

run_training() {
    # Check if training already completed
    if [ -f "$MODELS_DIR/exp4_best.pth" ] && [ -f "$RESULTS_DIR/exp4_results.json" ]; then
        log_step "Training skipped - exp4_best.pth and exp4_results.json already exist"
        return 0
    fi
    
    log_step "Starting training for experiment $EXPERIMENT_NO with random positioning and goal rewards"
    log_step "Logging training output to: $RUN_LOG_DIR/training.log"
    
    cd "$SCRIPTS_DIR/experiment4"
    python train.py --log_dir "$RUN_LOG_DIR/training.log" > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Training completed"
    
    # Log training summary if available
    if [ -f "$RUN_LOG_DIR/training.log" ]; then
        log_step "Training summary:"
        grep -i "epoch\|accuracy\|loss\|random" "$RUN_LOG_DIR/training.log" | tail -10 || true
    fi
}

run_evaluation() {
    # Check if evaluation already completed
    if [ -f "$RESULTS_DIR/cross_species_evaluation_exp4.json" ]; then
        log_step "Evaluation skipped - cross_species_evaluation_exp4.json already exists"
        return 0
    fi
    
    log_step "Starting evaluation for experiment $EXPERIMENT_NO"
    log_step "Logging evaluation output to: $RUN_LOG_DIR/evaluation.log"
    
    cd "$SCRIPTS_DIR/experiment4"
    python evaluate.py --config_override --n_past_eval --analysis_only --embeddings > "$RUN_LOG_DIR/evaluation.log" 2>&1
    
    log_step "Evaluation completed"
    
    # Log evaluation summary if available
    if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
        log_step "Evaluation summary:"
        grep -i "accuracy\|performance\|random" "$RUN_LOG_DIR/evaluation.log" | tail -5 || true
    fi
}

run_visualization() {
    # Check if visualization already completed
    if [ -f "$PLOTS_DIR/training_curves_exp4.png" ] && [ -f "$PLOTS_DIR/confusion_matrix_exp4.png" ]; then
        log_step "Visualization skipped - training_curves_exp4.png and confusion_matrix_exp4.png already exist"
        return 0
    fi
    
    log_step "Starting visualization for experiment $EXPERIMENT_NO"
    log_step "Logging visualization output to: $RUN_LOG_DIR/visualization.log"
    
    cd "$SCRIPTS_DIR/experiment4"
    python visualize.py --plot_type "all" > "$RUN_LOG_DIR/visualization.log" 2>&1
    
    log_step "Visualization completed"
}

# Additional function to check random positioning and reward implementation
check_random_implementation() {
    log_step "Checking random positioning and goal rewards implementation..."
    
    cd "$SCRIPTS_DIR/experiment4"
    
    # Check if config has random parameters
    if grep -q "random_positions.*True" config.py; then
        log_step "✓ Random positions enabled in config.py"
    else
        log_step "✗ Random positions not enabled in config.py"
    fi
    
    if grep -q "random_goal_rewards.*True" config.py; then
        log_step "✓ Random goal rewards enabled in config.py"
    else
        log_step "✗ Random goal rewards not enabled in config.py"
    fi
    
    # Check if environment supports random features
    if grep -q "random" environment.py 2>/dev/null; then
        log_step "✓ Random features found in environment.py"
    else
        log_step "✗ Random features not found in environment.py"
    fi
    
    # Check if data_generation.py exists
    if [ -f "data_generation.py" ]; then
        log_step "✓ data_generation.py exists"
    else
        log_step "✗ data_generation.py not found"
    fi
    
    log_step "Random implementation check completed"
}

# Main execution
case $COMMAND in
    data_generation)
        check_random_implementation
        run_data_generation
        ;;
    train)
        run_training
        ;;
    evaluate)
        run_evaluation
        ;;
    visualize)
        run_evaluation
        run_visualization
        ;;
    all)
        log_step "Running complete ToMnetF pipeline for experiment $EXPERIMENT_NO with random positioning and goal rewards"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        check_random_implementation
        run_data_generation
        run_training
        run_evaluation
        run_visualization
        log_step "Complete pipeline finished successfully"
        ;;
    check)
        check_random_implementation
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
echo "Experiment 4 Random Features:"
echo "  - Random player positions create diverse starting scenarios"
echo "  - Random goal rewards provide varied incentive structures"
echo "  - Enhanced environment variability for robust ToMnet training"
echo "  - Evaluation includes analysis of performance across different configurations"