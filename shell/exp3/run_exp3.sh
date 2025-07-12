#!/bin/bash
# nohup bash shell/exp3/run_exp3.sh all > exp3.log 2>&1 &
# Complete workflow automation for KeyDoor experiment 3 with ToMnet training
# Usage: bash run_exp3.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=3
VALIDATION_GAMES=2000
TEST_RANDOM_SEED=123
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/script/exp3"
TRAIN_DATA_DIR="$BASE_DIR/data/exp3"
TEST_DATA_DIR="$BASE_DIR/data/exp3/test"
RESULTS_DIR="$BASE_DIR/results/exp3"
LOG_DIR="$BASE_DIR/log/exp3"

# Create timestamp for this run
if [ -n "$2" ]; then
    TIMESTAMP="$2"
else
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
fi
RUN_LOG_DIR="$LOG_DIR/$TIMESTAMP"
RESULTS_DIR="$RESULTS_DIR/$TIMESTAMP"

# Create directories
mkdir -p "$TRAIN_DATA_DIR" "$TEST_DATA_DIR" "$RESULTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"

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
    echo "  data_generation       Generate KeyDoor trajectory data for training"
    echo "  test_data_generation  Generate test data ($VALIDATION_GAMES games with seed $TEST_RANDOM_SEED) for evaluation"
    echo "  train                Train ToMnet model for KeyDoor experiment"
    echo "  evaluate             Evaluate trained model performance"
    echo "  visualize            Create plots and visualizations"
    echo "  all                  Run complete pipeline including test data generation"
    echo ""
    echo "KeyDoor Experiment 3 Features:"
    echo "  - Multi-colored key-door environment (4 keys, 4 doors)"
    echo "  - A* agent for optimal trajectory generation"
    echo "  - Automatic key pickup and door opening mechanics"
    echo "  - Successor representation labels for ToMnet training"
    echo "  - ToMnet architecture adapted for KeyDoor environment"
    echo "  - Character embedding with past episode generation"
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
    
    log_step "Starting KeyDoor trajectory generation for experiment $EXPERIMENT_NO"
    log_step "Logging data generation output to: $RUN_LOG_DIR/train_data_generation.log"
    
    cd "$SCRIPTS_DIR"
    python generate.py --config_override --save_dir "$TRAIN_DATA_DIR" > "$RUN_LOG_DIR/train_data_generation.log" 2>&1
    
    log_step "Data generation completed"
    
    # Log generation summary if available
    if [ -f "$RUN_LOG_DIR/train_data_generation.log" ]; then
        log_step "Data generation summary:"
        grep -i "generated.*games\|success\|completed" "$RUN_LOG_DIR/train_data_generation.log" | tail -5 || true
    fi
    
    # Count generated files
    local generated_files=$(find "$TRAIN_DATA_DIR" -name "test*.txt" | wc -l)
    log_step "Generated $generated_files trajectory files in $TRAIN_DATA_DIR"
}

run_test_data_generation() {
    # Check if test data already exists
    if [ -f "$TEST_DATA_DIR/test1.txt" ]; then
        log_step "Test data generation skipped - $VALIDATION_GAMES test files already exist in $TEST_DATA_DIR"
        return 0
    fi
    
    log_step "Starting test data generation for experiment $EXPERIMENT_NO"
    log_step "Generating $VALIDATION_GAMES test games with random seed $TEST_RANDOM_SEED"
    log_step "Logging test data generation output to: $RUN_LOG_DIR/test_data_generation.log"
    
    # Create test data directory
    mkdir -p "$TEST_DATA_DIR"
    
    cd "$SCRIPTS_DIR"
    python generate.py --config_override --n_games "$VALIDATION_GAMES" --save_dir "$TEST_DATA_DIR" --random_seed "$TEST_RANDOM_SEED" > "$RUN_LOG_DIR/test_data_generation.log" 2>&1

    # Verify test data was generated correctly
    GENERATED_TEST_FILES=$(find "$TEST_DATA_DIR" -name "test*.txt" | wc -l)
    if [ "$GENERATED_TEST_FILES" -eq "$VALIDATION_GAMES" ]; then
        log_step "Test data generation completed successfully - $GENERATED_TEST_FILES files generated"
    else
        log_step "Warning: Expected $VALIDATION_GAMES test files, but found $GENERATED_TEST_FILES"
    fi
}

run_training() {
    # Check if training already completed
    if ls "$RESULTS_DIR"/best_model.pth 1> /dev/null 2>&1; then
        log_step "Training skipped - best_model.pth already exists"
        return 0
    fi
    
    log_step "Starting ToMnet training for experiment $EXPERIMENT_NO"
    log_step "Logging training output to: $RUN_LOG_DIR/training.log"
    
    cd "$SCRIPTS_DIR"
    python train.py --config_override --data_dir "$TRAIN_DATA_DIR" --save_dir "$RESULTS_DIR" > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Training completed"
    
    # Log training summary if available
    if [ -f "$RUN_LOG_DIR/training.log" ]; then
        log_step "Training summary:"
        grep -i "epoch\|best.*loss\|accuracy\|completed" "$RUN_LOG_DIR/training.log" | tail -10 || true
    fi
}

run_evaluation() {
    # Check if evaluation script exists
    if [ ! -f "$SCRIPTS_DIR/evaluate.py" ]; then
        log_step "Evaluation skipped - evaluate.py not found"
        log_step "Create evaluate.py script for model evaluation"
        return 0
    fi
    
    # Check if evaluation already completed
    if [ -f "$RESULTS_DIR/evaluation_results.json" ]; then
        log_step "Evaluation skipped - evaluation_results.json already exists"
        return 0
    fi
    
    log_step "Starting evaluation for experiment $EXPERIMENT_NO"
    log_step "Logging evaluation output to: $RUN_LOG_DIR/evaluation.log"
    
    cd "$SCRIPTS_DIR"
    python evaluate.py --config_override --test_data_dir "$TEST_DATA_DIR" --result_dir "$RESULTS_DIR" --model_path "$RESULTS_DIR/best_model.pth" --save_predictions --plot_type "all" > "$RUN_LOG_DIR/evaluation.log" 2>&1

    # python script/exp3/evaluate.py --config_override --test_data_dir "./data/exp3/test" --result_dir "./results/exp3/20250711_192952" --model_path "./results/exp3/20250711_192952/best_model.pth" --save_predictions --plot_type "all"

    log_step "Evaluation completed"
    
    # Log evaluation summary if available
    if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
        log_step "Evaluation summary:"
        grep -i "accuracy\|performance\|results" "$RUN_LOG_DIR/evaluation.log" | tail -5 || true
    fi
}


run_visualization() {
    # Check if visualization script exists
    if [ ! -f "$SCRIPTS_DIR/visualize.py" ]; then
        log_step "Visualization skipped - visualize.py not found"
        log_step "Create visualize.py script for result visualization"
        return 0
    fi
    
    # Check if visualization already completed (check for plot directory)
    if [ -d "$RESULTS_DIR/plots" ] && [ "$(ls -A $RESULTS_DIR/plots/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
        log_step "Visualization skipped - plots already exist in $RESULTS_DIR/plots"
        return 0
    fi
    
    log_step "Starting visualization for experiment $EXPERIMENT_NO"
    log_step "Logging visualization output to: $RUN_LOG_DIR/visualization.log"
    
    cd "$SCRIPTS_DIR"
    python visualize.py --config_override --result_dir "$RESULTS_DIR" --plot_dir "$RESULTS_DIR/plots" --plot_type "all"  > "$RUN_LOG_DIR/visualization.log" 2>&1

    # python script/exp3/visualize.py --config_override --result_dir "./results/exp3/20250712_035356"  --plot_dir "./results/exp3/20250712_035356/plots" --plot_type "all"

    log_step "Visualization completed"
    
    # Log visualization summary if available
    if [ -f "$RUN_LOG_DIR/visualization.log" ]; then
        log_step "Visualization summary:"
        grep -i "completed\|saved\|error" "$RUN_LOG_DIR/visualization.log" | tail -5 || true
    fi
}

# Function to check KeyDoor experiment implementation
check_exp3_implementation() {
    log_step "Checking KeyDoor experiment implementation..."
    
    cd "$SCRIPTS_DIR"
    
    # Check if required files exist
    if [ -f "config.py" ]; then
        log_step "✓ config.py exists"
    else
        log_step "✗ config.py not found"
    fi
    
    if [ -f "generate.py" ]; then
        log_step "✓ generate.py exists"
    else
        log_step "✗ generate.py not found"
    fi
    
    if [ -f "train.py" ]; then
        log_step "✓ train.py exists"
    else
        log_step "✗ train.py not found"
    fi
    
    if [ -f "tomnet.py" ]; then
        log_step "✓ tomnet.py exists"
    else
        log_step "✗ tomnet.py not found"
    fi
    
    if [ -f "data_generation.py" ]; then
        log_step "✓ data_generation.py exists"
    else
        log_step "✗ data_generation.py not found"
    fi
    
    # Check if config has KeyDoor-specific parameters
    if grep -q "KeyDoor" config.py; then
        log_step "✓ KeyDoor environment configured in config.py"
    else
        log_step "✗ KeyDoor environment not configured in config.py"
    fi
    
    if grep -q "astar" config.py; then
        log_step "✓ A* agent configured in config.py"
    else
        log_step "✗ A* agent not configured in config.py"
    fi
    
    # Check if ToMnet model supports KeyDoor
    if grep -q "action_space.*7" tomnet.py 2>/dev/null; then
        log_step "✓ KeyDoor action space (7 actions) configured in tomnet.py"
    else
        log_step "✗ KeyDoor action space not configured in tomnet.py"
    fi
    
    if grep -q "goal_space.*4" tomnet.py 2>/dev/null; then
        log_step "✓ KeyDoor goal space (4 goals) configured in tomnet.py"
    else
        log_step "✗ KeyDoor goal space not configured in tomnet.py"
    fi
    
    log_step "KeyDoor experiment implementation check completed"
}

# Main execution
case $COMMAND in
    data_generation)
        check_exp3_implementation
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
        log_step "Running complete KeyDoor pipeline for experiment $EXPERIMENT_NO"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        check_exp3_implementation
        run_data_generation
        run_test_data_generation
        run_training
        run_evaluation
        run_visualization
        log_step "Complete pipeline finished successfully"
        ;;
    check)
        check_exp3_implementation
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
echo "KeyDoor Experiment 3 Features:"
echo "  - Multi-colored key-door environment (4 keys, 4 doors)"
echo "  - A* agent for optimal trajectory generation"
echo "  - Automatic key pickup and door opening mechanics"
echo "  - Successor representation labels for ToMnet training"
echo "  - ToMnet architecture adapted for KeyDoor environment"
echo "  - Character embedding with past episode generation from batch"
echo "  - Early stopping and model checkpointing"
echo "  - Comprehensive training history tracking and visualization"