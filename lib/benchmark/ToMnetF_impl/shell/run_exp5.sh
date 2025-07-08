#!/bin/bash
# nohup bash shell/run_exp5.sh all > experiment5.log 2>&1 &
# Complete workflow automation for ToMnetF experiment5 with n_past character embedding and rank matching
# Usage: bash run_exp5.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=5
VALIDATION_GAMES=2000
TEST_RANDOM_SEED=123
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
TRAIN_DATA_DIR="$BASE_DIR/data/experiment5"
TEST_DATA_DIR="$BASE_DIR/data/experiment5/test"
MODELS_DIR="$BASE_DIR/models/experiment5"
RESULTS_DIR="$BASE_DIR/result/experiment5"
PLOTS_DIR="$BASE_DIR/plots/experiment5"
LOG_DIR="$BASE_DIR/log"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$LOG_DIR/experiment5/$TIMESTAMP"

# Create directories
mkdir -p "$TRAIN_DATA_DIR" "$TEST_DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"

# Function to pre-create all log files
create_log_files() {
    local log_files=(
        "execution.log"
        "train_data_generation.log"
        "test_data_generation.log"
        "training.log"
        "evaluation.log"
        "visualization.log"
    )
    
    log_step "Pre-creating log files..."
    for log_file in "${log_files[@]}"; do
        touch "$RUN_LOG_DIR/$log_file"
        log_step "Created: $RUN_LOG_DIR/$log_file"
    done
    log_step "All log files pre-created successfully"
}

# Parse command line arguments
COMMAND=${1:-all}

print_usage() {
    echo "Usage: $0 [data_generation|test_data_generation|train|evaluate|visualize|all]"
    echo ""
    echo "Commands:"
    echo "  data_generation       Generate trajectory data with n_past character embedding"
    echo "  test_data_generation  Generate test data (2000 games with seed 123) for evaluation"
    echo "  train                Train ToMnet model for experiment 5"
    echo "  evaluate             Evaluate trained model"
    echo "  visualize            Create plots and visualizations including n_past and rank matching analysis"
    echo "  all                  Run complete pipeline including test data generation"
    echo ""
    echo "Experiment 5 Features:"
    echo "  - N_past character embedding (n_past_min=0, n_past_max=4)"
    echo "  - Goal rank matching with configurable threshold (rank_threshold=4)"
    echo "  - Random goal rewards enabled (random_goal_rewards=True)"
    echo "  - Enhanced character understanding through historical episodes"
    echo "  - Test data generation matches validation set size (2000 games)"
    echo ""
}

log_step() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message"
    echo "$message" >> "$RUN_LOG_DIR/execution.log"
}

# Pre-create all log files after log_step function is defined
create_log_files

run_data_generation() {
    # Check if data already exists
    if [ -f "$TRAIN_DATA_DIR/test1.txt" ]; then
        log_step "Data generation skipped - test*.txt already exists"
        return 0
    fi
    
    log_step "Starting data generation for experiment $EXPERIMENT_NO with n_past character embedding"
    log_step "Logging data generation output to: $RUN_LOG_DIR/train_data_generation.log"
    
    cd "$SCRIPTS_DIR/experiment5"
    python generate.py --config_override --save_dir "$TRAIN_DATA_DIR" > "$RUN_LOG_DIR/train_data_generation.log" 2>&1
    
    log_step "Data generation completed"
    
    # Log n_past and rank matching statistics if available
    if [ -f "$RUN_LOG_DIR/train_data_generation.log" ]; then
        log_step "N_past character embedding and rank matching summary:"
        grep -i "n_past\|rank\|character\|embedding" "$RUN_LOG_DIR/train_data_generation.log" | tail -5 || true
    fi
}

run_training() {
    # Check if training already completed
    if [ -f "$MODELS_DIR/exp5_best.pth" ] && [ -f "$RESULTS_DIR/exp5_results.json" ]; then
        log_step "Training skipped - exp5_best.pth and exp5_results.json already exist"
        return 0
    fi
    
    log_step "Starting training for experiment $EXPERIMENT_NO with n_past character embedding and rank matching"
    log_step "Logging training output to: $RUN_LOG_DIR/training.log"
    
    cd "$SCRIPTS_DIR/experiment5"
    python train.py --log_dir "$RUN_LOG_DIR/training.log" > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Training completed"
    
    # Log training summary if available
    if [ -f "$RUN_LOG_DIR/training.log" ]; then
        log_step "Training summary:"
        grep -i "epoch\|accuracy\|loss\|n_past\|rank" "$RUN_LOG_DIR/training.log" | tail -10 || true
    fi
}

run_evaluation() {
    # Check if evaluation already completed
    if [ -f "$RESULTS_DIR/cross_species_evaluation_exp5.json" ]; then
        log_step "Evaluation skipped - cross_species_evaluation_exp5.json already exists"
        return 0
    fi
    
    log_step "Starting evaluation for experiment $EXPERIMENT_NO"
    log_step "Logging evaluation output to: $RUN_LOG_DIR/evaluation.log"
    
    cd "$SCRIPTS_DIR/experiment5"
    python evaluate.py --config_override > "$RUN_LOG_DIR/evaluation.log" 2>&1
    
    log_step "Evaluation completed"
    
    # Log evaluation summary if available
    if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
        log_step "Evaluation summary:"
        grep -i "accuracy\|performance\|n_past\|rank" "$RUN_LOG_DIR/evaluation.log" | tail -5 || true
    fi
}

run_test_data_generation() {
    # Calculate number of test games needed (same as validation data: 10% of training data)
    # From config.py: n_games=20000, training_proportion=0.9, so validation=2000 games
    
    # Check if test data already exists and has correct number of files
    if [ -f "$TEST_DATA_DIR/test1.txt" ]; then
        log_step "Test data generation skipped - $VALIDATION_GAMES test files already exist in $TEST_DATA_DIR"
        return 0
    fi
    
    log_step "Starting test data generation for experiment $EXPERIMENT_NO"
    log_step "Generating $VALIDATION_GAMES test games with random seed 123"
    log_step "Logging test data generation output to: $RUN_LOG_DIR/test_data_generation.log"
    
    # Create test data directory
    mkdir -p "$TEST_DATA_DIR"
    
    cd "$SCRIPTS_DIR/experiment5"
    python generate.py --config_override --n_games "$VALIDATION_GAMES" --random_seed "$TEST_RANDOM_SEED" --save_dir "$TEST_DATA_DIR" > "$RUN_LOG_DIR/test_data_generation.log" 2>&1

    # Verify test data was generated correctly
    GENERATED_TEST_FILES=$(find "$TEST_DATA_DIR" -name "test*.txt" | wc -l)
    if [ "$GENERATED_TEST_FILES" -eq "$VALIDATION_GAMES" ]; then
        log_step "Test data generation completed successfully - $GENERATED_TEST_FILES files generated"
    else
        log_step "Warning: Expected $VALIDATION_GAMES test files, but found $GENERATED_TEST_FILES"
    fi
}

run_visualization() {
    # Check if visualization already completed
    if [ -f "$PLOTS_DIR/training_curves_exp5.png" ] && [ -f "$PLOTS_DIR/confusion_matrix_exp5.png" ]; then
        log_step "Visualization skipped - training_curves_exp5.png and confusion_matrix_exp5.png already exist"
        return 0
    fi
    
    log_step "Starting visualization for experiment $EXPERIMENT_NO"
    log_step "Logging visualization output to: $RUN_LOG_DIR/visualization.log"
    
    cd "$SCRIPTS_DIR/experiment5"
    python visualize.py --plot_type "all" > "$RUN_LOG_DIR/visualization.log" 2>&1
    
    log_step "Visualization completed"
}

# Additional function to check n_past and rank matching implementation
check_exp5_implementation() {
    log_step "Checking n_past character embedding and rank matching implementation..."
    
    cd "$SCRIPTS_DIR/experiment5"
    
    # Check if config has n_past parameters
    if grep -q "use_n_past.*True" config.py; then
        log_step "✓ N_past character embedding enabled in config.py"
    else
        log_step "✗ N_past character embedding not enabled in config.py"
    fi
    
    if grep -q "n_past_max.*4" config.py; then
        log_step "✓ N_past max set to 4 in config.py"
    else
        log_step "✗ N_past max not set to 4 in config.py"
    fi
    
    if grep -q "RANK_THRESHOLD.*4" config.py; then
        log_step "✓ Rank threshold set to 4 in config.py"
    else
        log_step "✗ Rank threshold not set to 4 in config.py"
    fi
    
    if grep -q "random_goal_rewards.*True" config.py; then
        log_step "✓ Random goal rewards enabled in config.py"
    else
        log_step "✗ Random goal rewards not enabled in config.py"
    fi
    
    # Check if ToMnet model supports n_past
    if grep -q "n_past\|past" tomnet.py 2>/dev/null; then
        log_step "✓ N_past features found in tomnet.py"
    else
        log_step "✗ N_past features not found in tomnet.py"
    fi
    
    # Check if data_generation.py exists
    if [ -f "data_generation.py" ]; then
        log_step "✓ data_generation.py exists"
    else
        log_step "✗ data_generation.py not found"
    fi
    
    log_step "Experiment 5 implementation check completed"
}

# Main execution
case $COMMAND in
    data_generation)
        check_exp5_implementation
        run_data_generation
        ;;
    test_data_generation)
        run_test_data_generation
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
        log_step "Running complete ToMnetF pipeline for experiment $EXPERIMENT_NO with n_past character embedding and rank matching"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        check_exp5_implementation
        run_data_generation
        run_test_data_generation
        run_training
        run_evaluation
        run_visualization
        log_step "Complete pipeline finished successfully"
        ;;
    check)
        check_exp5_implementation
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

# List all log files with sizes
for log_file in "$RUN_LOG_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        filename=$(basename "$log_file")
        if [ "$filename" != "execution.log" ]; then
            size=$(ls -lh "$log_file" | awk '{print $5}')
            echo "  - $log_file (size: $size)"
        fi
    fi
done

echo ""
echo "Experiment 5 Features:"
echo "  - N_past character embedding (0-4 episodes) for enhanced character understanding"
echo "  - Goal rank matching with configurable threshold (rank_threshold=4)"
echo "  - Random goal rewards create varied incentive structures"
echo "  - Historical episode analysis for improved ToMnet performance"
echo "  - Evaluation includes analysis of n_past effectiveness and rank matching accuracy"