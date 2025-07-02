#!/bin/bash
# nohup bash shell/run_exp5.sh all > experiment.log 2>&1 &
# Main launcher script for ToMnet Figure 5 experiments
# 
# This script provides a convenient way to run the complete Figure 5 workflow
# from data generation to visualization with goal-directed agents.
#
# Usage:
#   bash run_exp5.sh [COMMAND] [OPTIONS]

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
ToMnet Figure 5 Experiment Launcher

DESCRIPTION:
    Complete workflow for running ToMnet Figure 5 experiments with goal-directed agents
    and generating Figure 5b,d visualizations.

USAGE:
    bash run_exp5.sh [COMMAND] [OPTIONS]

COMMANDS:
    generate [OPTIONS]      Generate goal-directed agent training data
    train [OPTIONS]         Train ToMnet models on goal-directed data
    evaluate [OPTIONS]      Run evaluation for Figure 5b,d data
    visualize [OPTIONS]     Generate Figure 5 visualizations
    all [OPTIONS]           Run complete workflow (generate + train + evaluate + visualize)
    clean                   Clean up generated files
    help                    Show this help message

EXAMPLES:
    # Complete workflow with default settings
    bash run_exp5.sh all
    
    # Generate data only
    bash run_exp5.sh generate --n_agents 200 --n_episodes_per_agent 100
    
    # Train models with custom parameters
    bash run_exp5.sh train --n_epochs 50 --batch_size 64
    
    # Generate visualizations and save plots
    bash run_exp5.sh visualize --save
    
    # Clean up all generated files
    bash run_exp5.sh clean

WORKFLOW:
    1. Data Generation: Creates goal-directed agent training data
    2. Training: Trains ToMnet models on goal-directed agent data  
    3. Evaluation: Runs evaluation to compute Figure 5b,d data
    4. Visualization: Generates Figure 5b,d plots

FILES ORGANIZATION:
    scripts/            - Python source files (.py)
    shell/              - Shell scripts (.sh)
    data/figure5/       - Generated training data
    models/figure5/     - Trained model checkpoints
    result/figure5/     - Evaluation results
    plots/figure5/      - Generated visualization plots

EOF
}

# Function to run data generation
run_data_generation() {
    # Check for help flag
    for arg in "$@"; do
        if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
            python scripts/generate_figure5_data.py --help
            return 0
        fi
    done
    
    # Check if data already exists
    if ls ./data/figure5/*.pkl 1> /dev/null 2>&1; then
        print_warning "Training data already exists in ./data/figure5/"
        print_info "Skipping data generation step. Remove ./data/figure5/*.pkl to re-generate."
        return 0
    fi
    
    print_info "Starting Figure 5 data generation..."
    
    # Create log directory with timestamp
    LOG_DIR="log/data_generation/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Run data generation in background with logging
    echo "Starting data generation at $(date)" > "$LOG_DIR/execution.log"
    python scripts/generate_figure5_data.py --n_agents 100 --n_episodes_per_agent 100 --alpha_reward 0.01 "$@" >> "$LOG_DIR/execution.log" 2>&1 &
    GEN_PID=$!
    echo $GEN_PID > "$LOG_DIR/process.pid"
    print_info "Data generation started in background with PID: $GEN_PID"
    print_info "Logs will be written to: $LOG_DIR/execution.log"
    
    # Wait for data generation to complete
    wait $GEN_PID
    if [ $? -eq 0 ]; then
        print_success "Data generation completed!"
    else
        print_error "Data generation failed! Check log: $LOG_DIR/execution.log"
        exit 1
    fi
}

# Function to run training
run_training() {
    # Check for help flag
    for arg in "$@"; do
        if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
            python scripts/train_figure5.py --help
            return 0
        fi
    done
    
    # Check if training results already exist
    if ls ./models/figure5/*.pth 1> /dev/null 2>&1; then
        print_warning "Training results already exist in ./models/figure5/"
        print_info "Skipping training step. Remove ./models/figure5/*.pth to re-train."
        return 0
    fi
    
    print_info "Starting Figure 5 ToMnet training..."
    
    # Create log directory with timestamp
    LOG_DIR="log/training/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Run training in background with logging
    echo "Starting training at $(date)" > "$LOG_DIR/execution.log"
    python scripts/train_figure5.py --n_agents 100 --n_episodes_per_agent 100 --n_epochs 100 --alpha_reward 0.01 "$@" >> "$LOG_DIR/execution.log" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$LOG_DIR/process.pid"
    print_info "Training started in background with PID: $TRAIN_PID"
    print_info "Logs will be written to: $LOG_DIR/execution.log"
    
    # Wait for training to complete
    wait $TRAIN_PID
    if [ $? -eq 0 ]; then
        print_success "Training completed!"
    else
        print_error "Training failed! Check log: $LOG_DIR/execution.log"
        exit 1
    fi
}

# Function to run evaluation
run_evaluation() {
    # Check for help flag
    for arg in "$@"; do
        if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
            python scripts/evaluate_figure5.py --help
            return 0
        fi
    done
    
    # Check if evaluation results already exist
    if [ -f "./result/figure5/figure5_results.pkl" ]; then
        print_warning "Evaluation results already exist at ./result/figure5/figure5_results.pkl"
        print_info "Skipping evaluation step. Remove ./result/figure5/figure5_results.pkl to re-evaluate."
        return 0
    fi
    
    print_info "Starting Figure 5 evaluation..."
    
    # Create log directory with timestamp
    LOG_DIR="log/evaluation/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Run evaluation in background with logging
    echo "Starting evaluation at $(date)" > "$LOG_DIR/execution.log"
    python scripts/evaluate_figure5.py --model_dir models/figure5 --data_dir data/figure5 --output_dir result/figure5 "$@" >> "$LOG_DIR/execution.log" 2>&1 &
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
}

# Function to run visualization
run_visualization() {
    # Check for help flag
    for arg in "$@"; do
        if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
            python scripts/visualize_figure5.py --help
            return 0
        fi
    done
    
    # Check if visualization results already exist
    if ls ./plots/figure5/* 1> /dev/null 2>&1; then
        print_warning "Visualization results already exist in ./plots/figure5/"
        print_info "Skipping visualization step. Remove ./plots/figure5/* to re-generate visualizations."
        return 0
    fi
    
    print_info "Starting Figure 5 visualization..."
    
    # Create log directory with timestamp
    LOG_DIR="log/visualization/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Run visualization in background with logging
    echo "Starting visualization at $(date)" > "$LOG_DIR/execution.log"
    
    # Check if visualize_figure5.sh exists, otherwise try running the python script directly
    if [ -f "shell/visualize_figure5.sh" ]; then
        bash shell/visualize_figure5.sh --save "$@" >> "$LOG_DIR/execution.log" 2>&1 &
    elif [ -f "scripts/visualize_figure5.py" ]; then
        print_info "Running visualize_figure5.py directly..."
        python scripts/visualize_figure5.py --save_plots "$@" >> "$LOG_DIR/execution.log" 2>&1 &
    else
        print_error "No visualization script found!"
        exit 1
    fi
    
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
    print_warning "This will delete all generated data, models, and results for Figure 5."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up generated files..."
        rm -rf data/figure5/ models/figure5/ result/figure5/ plots/figure5/
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
    generate)
        run_data_generation "$@"
        ;;
    train)
        run_training "$@"
        ;;
    evaluate)
        run_evaluation "$@"
        ;;
    visualize)
        run_visualization "$@"
        ;;
    all)
        print_info "Running complete Figure 5 workflow..."
        echo ""
        run_data_generation "$@"
        echo ""
        run_training "$@"
        echo ""
        run_evaluation "$@"
        echo ""
        run_visualization --save
        echo ""
        print_success "Complete Figure 5 workflow finished!"
        ;;
    clean)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo "Use 'bash run_exp5.sh help' for usage information"
        exit 1
        ;;
esac