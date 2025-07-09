#!/bin/bash
# nohup bash shell/exp3/rrun_exp3.sh > exp3.log 2>&1 &
# KeyDoor trajectory generation script for experiment 3
# Usage: bash run_generate.sh [n_games] [agent_type] [env_size]

set -e  # Exit on error

# Configuration
EXPERIMENT_NO=3
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$BASE_DIR/script/exp3"
DATA_DIR="$BASE_DIR/data/exp3"
LOG_DIR="$BASE_DIR/log/exp3"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$LOG_DIR/$TIMESTAMP"

# Create directories
mkdir -p "$DATA_DIR" "$RUN_LOG_DIR"

# Function to pre-create all log files
create_log_files() {
    local log_files=(
        "execution.log"
        "generation.log"
    )
    
    log_step "Pre-creating log files..."
    for log_file in "${log_files[@]}"; do
        touch "$RUN_LOG_DIR/$log_file"
        log_step "Created: $RUN_LOG_DIR/$log_file"
    done
    log_step "All log files pre-created successfully"
}

# Parse command line arguments
N_GAMES=${1:-5}
AGENT_TYPE=${2:-astar}
ENV_SIZE=${3:-9x9}
RANDOM_SEED=${4:-42}
N_PROCESSES=${5:-}

print_usage() {
    echo "Usage: $0 [n_games] [agent_type] [env_size] [random_seed] [n_processes]"
    echo ""
    echo "Parameters:"
    echo "  n_games       Number of games to generate (default: 5)"
    echo "  agent_type    Type of agent [astar|value|random] (default: astar)"
    echo "  env_size      Environment size [5x5|9x9|11x11] (default: 9x9)"
    echo "  random_seed   Random seed for generation (default: 42)"
    echo "  n_processes   Number of parallel processes (default: auto)"
    echo ""
    echo "Examples:"
    echo "  $0 10 astar 9x9            # Generate 10 games with A* agent"
    echo "  $0 50 value 9x9 123        # Generate 50 games with value agent, seed 123"
    echo "  $0 100 random 9x9 456 8    # Generate 100 games with random agent, 8 processes"
    echo ""
    echo "KeyDoor Environment Features:"
    echo "  - 4 colored keys (red, green, blue, yellow)"
    echo "  - 4 matching doors that require corresponding keys"
    echo "  - Agent must collect target key and open target door"
    echo "  - Automatic key pickup and door opening mechanics"
    echo "  - Successor representation (SR) labels for ToMnet training"
    echo ""
}

log_step() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message"
    echo "$message" >> "$RUN_LOG_DIR/execution.log"
}

# Pre-create all log files after log_step function is defined
create_log_files

run_generation() {
    log_step "Starting KeyDoor trajectory generation for experiment $EXPERIMENT_NO"
    log_step "Parameters: n_games=$N_GAMES, agent_type=$AGENT_TYPE, env_size=$ENV_SIZE"
    log_step "Random seed: $RANDOM_SEED, processes: ${N_PROCESSES:-auto}"
    log_step "Logging generation output to: $RUN_LOG_DIR/generation.log"
    
    cd "$SCRIPTS_DIR"
    
    # Build command with config_override
    local cmd="python generate.py --config_override"
    cmd="$cmd --n_games $N_GAMES"
    cmd="$cmd --agent_type $AGENT_TYPE"
    cmd="$cmd --env_size $ENV_SIZE"
    cmd="$cmd --save_dir $DATA_DIR"
    cmd="$cmd --random_seed $RANDOM_SEED"
    
    # Add n_processes if specified
    if [ -n "$N_PROCESSES" ]; then
        cmd="$cmd --n_processes $N_PROCESSES"
    fi
    
    log_step "Running command: $cmd"
    
    # Execute the command
    eval "$cmd" > "$RUN_LOG_DIR/generation.log" 2>&1
    
    log_step "Generation completed successfully"
    
    # Log generation summary if available
    if [ -f "$RUN_LOG_DIR/generation.log" ]; then
        log_step "Generation summary:"
        grep -i "generated.*games\|success\|completed" "$RUN_LOG_DIR/generation.log" | tail -5 || true
    fi
    
    # Count generated files
    local generated_files=$(find "$DATA_DIR" -name "test*.txt" | wc -l)
    log_step "Generated $generated_files trajectory files in $DATA_DIR"
}

# Check for help flags
case $1 in
    help|--help|-h)
        print_usage
        exit 0
        ;;
    *)
        ;;
esac

# Validate agent type
case $AGENT_TYPE in
    astar|value|random)
        ;;
    *)
        echo "Error: Invalid agent type '$AGENT_TYPE'. Must be one of: astar, value, random"
        echo ""
        print_usage
        exit 1
        ;;
esac

# Validate environment size
case $ENV_SIZE in
    5x5|9x9|11x11)
        ;;
    *)
        echo "Error: Invalid environment size '$ENV_SIZE'. Must be one of: 5x5, 9x9, 11x11"
        echo ""
        print_usage
        exit 1
        ;;
esac

# Validate n_games is a positive integer
if ! [[ "$N_GAMES" =~ ^[0-9]+$ ]] || [ "$N_GAMES" -le 0 ]; then
    echo "Error: n_games must be a positive integer, got '$N_GAMES'"
    echo ""
    print_usage
    exit 1
fi

# Main execution
log_step "Starting KeyDoor trajectory generation pipeline"
log_step "All logs will be saved to: $RUN_LOG_DIR/"

run_generation

log_step "Script completed successfully"
log_step "Log files saved to: $RUN_LOG_DIR/"
echo ""
echo "Log files created:"
echo "  - $RUN_LOG_DIR/execution.log (main script execution log)"
echo "  - $RUN_LOG_DIR/generation.log (trajectory generation log)"

# List log files with sizes
for log_file in "$RUN_LOG_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        filename=$(basename "$log_file")
        size=$(ls -lh "$log_file" | awk '{print $5}')
        echo "  - $log_file (size: $size)"
    fi
done

echo ""
echo "Generated trajectory files:"
find "$DATA_DIR" -name "test*.txt" -type f | head -10 | while read file; do
    echo "  - $file"
done

if [ $(find "$DATA_DIR" -name "test*.txt" -type f | wc -l) -gt 10 ]; then
    echo "  ... and $(expr $(find "$DATA_DIR" -name "test*.txt" -type f | wc -l) - 10) more files"
fi

echo ""
echo "KeyDoor Experiment 3 Features:"
echo "  - Multi-colored key-door environment (4 keys, 4 doors)"
echo "  - Three agent types: A* (optimal), Value (reinforcement learning), Random"
echo "  - Automatic key pickup and door opening mechanics"
echo "  - Successor representation labels for ToMnet training"
echo "  - Parallel trajectory generation for efficiency"
echo "  - Compatible with experiment 5 ToMnet format"