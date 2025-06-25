#!/bin/bash
# Shell script for ToMnet Figure 3 Visualization
# 
# This script provides a convenient interface for generating Figure 3 visualizations
# from ToMnet cross-species evaluation results.
#
# Usage:
#   bash visualize_figure3.sh [OPTIONS]
#
# Examples:
#   bash visualize_figure3.sh                    # Use default settings
#   bash visualize_figure3.sh --save            # Save plots to files
#   bash visualize_figure3.sh --help            # Show help

set -e  # Exit on error

# Default settings
RESULTS_PATH="result/figure3/figure3_cross_species_results.pkl"
SAVE_PLOTS=false
OUTPUT_DIR="plots"
SHOW_HELP=false
DEVICE="cpu"

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
ToMnet Figure 3 Visualization Script

DESCRIPTION:
    Generate visualizations for Figure 3 of the "Machine Theory of Mind" paper.
    This script creates plots showing:
    - (a) Action likelihood vs number of past observations  
    - (b) 2D character embeddings colored by most frequent action
    - (c) KL-divergence matrix for cross-species generalization
    - (d) Hierarchical inference on mixed species

USAGE:
    bash visualize_figure3.sh [OPTIONS]

OPTIONS:
    --results_path PATH     Path to evaluation results pickle file
                           (default: result/figure3/figure3_cross_species_results.pkl)
    
    --save                 Save plots to files instead of displaying them
    
    --output_dir DIR       Directory to save plots when using --save
                           (default: plots)
    
    --device DEVICE        Device to use for any computations (cpu/cuda/mps)
                           (default: cpu)
    
    --help, -h             Show this help message

EXAMPLES:
    # Basic usage with default settings
    bash visualize_figure3.sh
    
    # Save plots to files
    bash visualize_figure3.sh --save
    
    # Use custom results file and output directory
    bash visualize_figure3.sh --results_path my_results.pkl --save --output_dir my_plots
    
    # Show help
    bash visualize_figure3.sh --help

PREREQUISITES:
    1. Run training: python train.py --experiment figure3
    2. Run evaluation: bash result/figure3/run_cross_species_evaluation.sh
    3. Then run this visualization script

OUTPUT:
    - If --save is used: PNG files saved to output directory
    - Otherwise: Interactive plots displayed on screen

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results_path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --save)
            SAVE_PLOTS=true
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    show_help
    exit 0
fi

# Print script header
echo "========================================================================"
echo "                ToMnet Figure 3 Visualization Script"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not available. Please install Python 3.7+"
    exit 1
fi

# Check if results file exists
if [ ! -f "$RESULTS_PATH" ]; then
    print_error "Results file not found: $RESULTS_PATH"
    echo ""
    print_info "Make sure you have run the evaluation first:"
    echo "  1. Train models: python scripts/train.py --experiment figure3"
    echo "  2. Run evaluation: bash result/figure3/run_cross_species_evaluation.sh"
    echo "  3. Then run this visualization script"
    exit 1
fi

print_success "Found results file: $RESULTS_PATH"

# Prepare visualization command
PYTHON_CMD="python scripts/visualize_figure3.py --results_path \"$RESULTS_PATH\""

if [ "$SAVE_PLOTS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --save_plots --output_dir \"$OUTPUT_DIR\""
    print_info "Plots will be saved to: $OUTPUT_DIR"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    print_info "Created output directory: $OUTPUT_DIR"
else
    print_info "Plots will be displayed interactively"
fi

print_info "Device: $DEVICE"
print_info "Running visualization command..."
echo ""

# Run the Python visualization script
print_info "Executing: $PYTHON_CMD"
eval $PYTHON_CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo ""
    print_success "Visualization completed successfully!"
    
    if [ "$SAVE_PLOTS" = true ]; then
        echo ""
        print_info "Generated plots:"
        if [ -d "$OUTPUT_DIR" ]; then
            ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || print_warning "No PNG files found in $OUTPUT_DIR"
        fi
        echo ""
        print_info "You can view the plots using any image viewer:"
        echo "  - figure3a_action_likelihood.png: Action likelihood vs training alpha"
        echo "  - figure3b_character_embeddings.png: 2D character embedding visualization"  
        echo "  - figure3c_cross_species_kl.png: Cross-species KL divergence matrix"
        echo "  - figure3d_mixed_species.png: Mixed species training performance"
    fi
else
    print_error "Visualization failed with exit code $?"
    exit 1
fi

echo ""
echo "========================================================================"
print_success "Figure 3 visualization script completed!"
echo "========================================================================"