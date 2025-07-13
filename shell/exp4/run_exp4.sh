#!/bin/bash
# nohup bash shell/exp4/run_exp4.sh all > exp4.log 2>&1 &
# Complete workflow automation for AchieverBlocker experiment 4 with ToMnet training
# Usage: bash run_exp4.sh [data_generation|train|evaluate|visualize|all]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=4
VALIDATION_GAMES=2000  # Reduced from 2000 for testing
TEST_RANDOM_SEED=123
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Data paths will be dynamically determined from config.py
RESULTS_DIR="$BASE_DIR/results/exp4"
LOG_DIR="$BASE_DIR/log/exp4"

# Create timestamp for this run
if [ -n "$2" ]; then
    TIMESTAMP="$2"
else
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
fi
RUN_LOG_DIR="$LOG_DIR/$TIMESTAMP"
RESULTS_DIR="$RESULTS_DIR/$TIMESTAMP"

# Create directories (data directories will be created automatically by generate.py)
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$RUN_LOG_DIR"

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
    echo "  data_generation       Generate AchieverBlocker trajectory data for training"
    echo "  test_data_generation  Generate test data ($VALIDATION_GAMES games with seed $TEST_RANDOM_SEED) for evaluation"
    echo "  train                Train ToMnet model for AchieverBlocker experiment"
    echo "  evaluate             Evaluate trained model performance"
    echo "  visualize            Create plots and visualizations"
    echo "  all                  Run complete pipeline including test data generation"
    echo ""
    echo "AchieverBlocker Experiment 4 Features:"
    echo "  - Multi-agent environment (Achiever and Blocker agents)"
    echo "  - Level-k reasoning and goal inference"
    echo "  - Successor representation (SR) data for both agents"
    echo "  - Agent type prediction (achiever vs blocker)"
    echo "  - ToMnet architecture adapted for multi-agent environment"
    echo "  - Trajectory slicing for improved training efficiency"
    echo "  - Goal ranking system for past episode generation"
    echo "  - Config-based data paths: ./data/{env_name}/{agent_type}/"
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
    log_step "Starting AchieverBlocker trajectory generation for experiment $EXPERIMENT_NO"
    log_step "Using config-based data path generation"
    log_step "Logging data generation output to: $RUN_LOG_DIR/train_data_generation.log"
    
    # Run generate.py from base directory to maintain correct relative paths
    python script/exp4/generate.py --config_override > "$RUN_LOG_DIR/train_data_generation.log" 2>&1
    
    log_step "Data generation completed"
    
    # Log generation summary if available
    if [ -f "$RUN_LOG_DIR/train_data_generation.log" ]; then
        log_step "Data generation summary:"
        grep -i "generated.*games\|success\|completed\|saving.*to" "$RUN_LOG_DIR/train_data_generation.log" | tail -5 || true
    fi
    
    # Try to determine the actual path and count files
    local actual_data_dir=$(grep -o "Data saved to: .*" "$RUN_LOG_DIR/train_data_generation.log" | head -1 | cut -d' ' -f4)
    if [ -d "$actual_data_dir" ]; then
        local generated_files=$(find "$actual_data_dir" -name "test*.txt" | wc -l)
        log_step "Generated $generated_files trajectory files in $actual_data_dir"
    else
        log_step "Could not determine generated data directory from logs"
    fi
}

run_test_data_generation() {
    log_step "Starting test data generation for experiment $EXPERIMENT_NO"
    log_step "Generating $VALIDATION_GAMES test games with random seed $TEST_RANDOM_SEED"
    log_step "Using config-based test data path generation"
    log_step "Logging test data generation output to: $RUN_LOG_DIR/test_data_generation.log"
    
    # Run generate.py from base directory to maintain correct relative paths
    python script/exp4/generate.py --config_override --n_games "$VALIDATION_GAMES" --random_seed "$TEST_RANDOM_SEED" --test_data > "$RUN_LOG_DIR/test_data_generation.log" 2>&1

    # Try to determine the actual test data path from logs and verify files
    local actual_test_dir=$(grep -o "Data saved to: .*" "$RUN_LOG_DIR/test_data_generation.log" | head -1 | cut -d' ' -f4)
    if [ -d "$actual_test_dir" ]; then
        GENERATED_TEST_FILES=$(find "$actual_test_dir" -name "test*.txt" | wc -l)
        if [ "$GENERATED_TEST_FILES" -eq "$VALIDATION_GAMES" ]; then
            log_step "Test data generation completed successfully - $GENERATED_TEST_FILES files generated in $actual_test_dir"
        else
            log_step "Warning: Expected $VALIDATION_GAMES test files, but found $GENERATED_TEST_FILES in $actual_test_dir"
        fi
    else
        log_step "Could not determine generated test data directory from logs"
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
    
    # Run train.py from base directory to maintain correct relative paths
    python script/exp4/train.py --config_override --save_dir "$RESULTS_DIR" > "$RUN_LOG_DIR/training.log" 2>&1
    
    log_step "Training completed"
    
    # Log training summary if available
    if [ -f "$RUN_LOG_DIR/training.log" ]; then
        log_step "Training summary:"
        grep -i "epoch\|best.*loss\|accuracy\|completed" "$RUN_LOG_DIR/training.log" | tail -10 || true
    fi
}

run_evaluation() {
    # Check if evaluation script exists
    if [ ! -f "script/exp4/evaluate.py" ]; then
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
    
    # Run evaluate.py from base directory to maintain correct relative paths
    python script/exp4/evaluate.py --config_override --result_dir "$RESULTS_DIR" --model_path "$RESULTS_DIR/best_model.pth" --save_predictions --plot_type "all" > "$RUN_LOG_DIR/evaluation.log" 2>&1

    log_step "Evaluation completed"
    
    # Log evaluation summary if available
    if [ -f "$RUN_LOG_DIR/evaluation.log" ]; then
        log_step "Evaluation summary:"
        grep -i "accuracy\|performance\|results" "$RUN_LOG_DIR/evaluation.log" | tail -5 || true
    fi
}


run_visualization() {
    # Check if visualization script exists
    if [ ! -f "script/exp4/visualize.py" ]; then
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
    
    # Run visualize.py from base directory to maintain correct relative paths
    python script/exp4/visualize.py --config_override --result_dir "$RESULTS_DIR" --plot_dir "$RESULTS_DIR/plots" --plot_type "all"  > "$RUN_LOG_DIR/visualization.log" 2>&1

    log_step "Visualization completed"
    
    # Log visualization summary if available
    if [ -f "$RUN_LOG_DIR/visualization.log" ]; then
        log_step "Visualization summary:"
        grep -i "completed\|saved\|error" "$RUN_LOG_DIR/visualization.log" | tail -5 || true
    fi
}

# Function to check AchieverBlocker experiment implementation
check_exp4_implementation() {
    log_step "Checking AchieverBlocker experiment implementation..."
    
    # Check if required files exist
    if [ -f "script/exp4/config.py" ]; then
        log_step "✓ config.py exists"
    else
        log_step "✗ config.py not found"
    fi
    
    if [ -f "script/exp4/generate.py" ]; then
        log_step "✓ generate.py exists"
    else
        log_step "✗ generate.py not found"
    fi
    
    if [ -f "script/exp4/train.py" ]; then
        log_step "✓ train.py exists"
    else
        log_step "✗ train.py not found"
    fi
    
    if [ -f "script/exp4/tomnet.py" ]; then
        log_step "✓ tomnet.py exists"
    else
        log_step "✗ tomnet.py not found"
    fi
    
    if [ -f "script/exp4/data_generation.py" ]; then
        log_step "✓ data_generation.py exists"
    else
        log_step "✗ data_generation.py not found"
    fi
    
    # Check if config has AchieverBlocker-specific parameters
    if grep -q "AchieverBlocker" "script/exp4/config.py" 2>/dev/null; then
        log_step "✓ AchieverBlocker environment configured in config.py"
    else
        log_step "✗ AchieverBlocker environment not configured in config.py"
    fi
    
    # Check if ToMnet model supports multi-agent
    if grep -q "agent.*prediction\|fc3_agent" "script/exp4/tomnet.py" 2>/dev/null; then
        log_step "✓ Agent prediction configured in tomnet.py"
    else
        log_step "✗ Agent prediction not configured in tomnet.py"
    fi
    
    if grep -q "action_space.*[67]" "script/exp4/tomnet.py" 2>/dev/null; then
        log_step "✓ AchieverBlocker action space configured in tomnet.py"
    else
        log_step "✗ AchieverBlocker action space not configured in tomnet.py"
    fi
    
    if grep -q "goal_space.*4" "script/exp4/tomnet.py" 2>/dev/null; then
        log_step "✓ AchieverBlocker goal space (4 goals) configured in tomnet.py"
    else
        log_step "✗ AchieverBlocker goal space not configured in tomnet.py"
    fi
    
    # Check for multi-agent specific functions
    if grep -q "prepare_data_for_training_multiagent" "script/exp4/train.py" 2>/dev/null; then
        log_step "✓ Multi-agent data preparation configured in train.py"
    else
        log_step "✗ Multi-agent data preparation not configured in train.py"
    fi
    
    log_step "AchieverBlocker experiment implementation check completed"
}

# Main execution
case $COMMAND in
    data_generation)
        check_exp4_implementation
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
        log_step "Running complete AchieverBlocker pipeline for experiment $EXPERIMENT_NO"
        log_step "All logs will be saved to: $RUN_LOG_DIR/"
        check_exp4_implementation
        run_data_generation
        run_test_data_generation
        run_training
        run_evaluation
        run_visualization
        log_step "Complete pipeline finished successfully"
        ;;
    check)
        check_exp4_implementation
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
echo "AchieverBlocker Experiment 4 Features:"
echo "  - Multi-agent environment (Achiever and Blocker agents)"
echo "  - Level-k reasoning and goal inference"
echo "  - Successor representation (SR) data for both agents"
echo "  - Agent type prediction (achiever vs blocker)"
echo "  - ToMnet architecture adapted for multi-agent environment"
echo "  - Trajectory slicing for improved training efficiency"
echo "  - Goal ranking system for past episode generation"
echo "  - Early stopping and model checkpointing"
echo "  - Comprehensive training history tracking and visualization"
echo "  - Config-based data paths: ./data/{env_name}/{agent_type}/"