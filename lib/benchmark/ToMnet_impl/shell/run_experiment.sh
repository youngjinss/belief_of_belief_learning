#!/bin/bash
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
    python scripts/train.py --experiment figure3 "$@"
    print_success "Training completed!"
}

# Function to run evaluation
run_evaluation() {
    print_info "Starting cross-species evaluation..."
    if [ -f "result/figure3/run_cross_species_evaluation.sh" ]; then
        bash result/figure3/run_cross_species_evaluation.sh
        print_success "Evaluation completed!"
    else
        print_error "Evaluation script not found. Run training first."
        exit 1
    fi
}

# Function to run visualization
run_visualization() {
    print_info "Starting Figure 3 visualization..."
    bash shell/visualize_figure3.sh "$@"
    print_success "Visualization completed!"
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