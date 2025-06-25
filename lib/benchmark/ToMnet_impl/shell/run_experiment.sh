#!/bin/bash
# nohup bash shell/run_experiment.sh > experiment.log 2>&1 &
# Main launcher script for ToMnet experiments
# 
# This script provides a convenient way to run the complete ToMnet workflow
# from training to visualization.
#
# Usage:
#   bash run_experiment.sh [COMMAND] [OPTIONS]

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
ToMnet Experiment Launcher

DESCRIPTION:
    Complete workflow for running ToMnet experiments and generating Figure 3 visualizations.

USAGE:
    bash run_experiment.sh [COMMAND] [OPTIONS]

COMMANDS:
    train [OPTIONS]         Train ToMnet models
    evaluate [OPTIONS]      Run cross-species evaluation
    visualize [OPTIONS]     Generate Figure 3 visualizations
    all [OPTIONS]           Run complete workflow (train + evaluate + visualize)
    clean                   Clean up generated files
    help                    Show this help message

EXAMPLES:
    # Complete workflow with default settings
    bash run_experiment.sh all
    
    # Train models only
    bash run_experiment.sh train --n_agents 100 --n_epochs 50
    
    # Generate visualizations with custom output
    bash run_experiment.sh visualize --save --output_dir my_plots
    
    # Clean up all generated files
    bash run_experiment.sh clean

WORKFLOW:
    1. Training: Creates models for different alpha values
    2. Evaluation: Runs cross-species evaluation on trained models  
    3. Visualization: Generates Figure 3 plots from evaluation results

FILES ORGANIZATION:
    scripts/    - Python source files (.py)
    shell/      - Shell scripts (.sh)
    notebook/   - Jupyter notebooks (.ipynb)
    data/       - Generated training data
    models/     - Trained model checkpoints
    result/     - Evaluation results
    plots/      - Generated visualization plots

EOF
}

# Function to run training
run_training() {
    print_info "Starting ToMnet training..."
    
    # Create log directory with timestamp
    LOG_DIR="log/training/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    print_info "Created log directory: $LOG_DIR"
    
    # Run training in background with logging
    echo "Starting training at $(date)" > "$LOG_DIR/execution.log"
    python scripts/train.py --experiment figure3 "$@" >> "$LOG_DIR/execution.log" 2>&1 &
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

# Function to run visualization
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
        rm -rf data/ models/ result/ plots/
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
        run_training "$@"
        ;;
    evaluate)
        run_evaluation "$@"
        ;;
    visualize)
        run_visualization "$@"
        ;;
    all)
        print_info "Running complete ToMnet workflow..."
        echo ""
        run_training "$@"
        echo ""
        run_evaluation "$@"
        echo ""
        run_visualization --save
        echo ""
        print_success "Complete workflow finished!"
        ;;
    clean)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo "Use 'bash run_experiment.sh help' for usage information"
        exit 1
        ;;
esac