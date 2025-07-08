#!/bin/bash
# Script to display the ToMnet project structure

# Create log directory with timestamp
LOG_DIR="log/show_structure/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "                    ToMnet Project Structure"
echo "========================================================================"
echo ""

# Execute tree command in background with logging
echo "Starting structure display at $(date)" > "$LOG_DIR/execution.log"
{
    tree -I '__pycache__|*.pyc|.git' -a || find . -type f -name "*.py" -o -name "*.sh" -o -name "*.ipynb" -o -name "*.md" | head -20
} >> "$LOG_DIR/execution.log" 2>&1 &
STRUCT_PID=$!
echo $STRUCT_PID > "$LOG_DIR/process.pid"

# Wait for command to complete and display output
wait $STRUCT_PID
cat "$LOG_DIR/execution.log" | tail -n +2  # Skip the timestamp line

echo ""
echo "========================================================================"
echo "                        Quick Start Guide"
echo "========================================================================"
echo ""
echo "🚀 COMPLETE WORKFLOW:"
echo "   bash shell/run_experiment.sh all"
echo ""
echo "📋 INDIVIDUAL STEPS:"
echo "   1. Training:      python scripts/train.py --experiment figure3"
echo "   2. Evaluation:    bash result/figure3/run_cross_species_evaluation.sh"
echo "   3. Visualization: bash shell/visualize_figure3.sh --save"
echo ""
echo "📁 FILE ORGANIZATION:"
echo "   scripts/    - Python source files (.py)"
echo "   shell/      - Shell scripts (.sh)"  
echo "   notebook/   - Jupyter notebooks (.ipynb)"
echo "   data/       - Generated training data"
echo "   models/     - Trained model checkpoints"
echo "   result/     - Evaluation results"
echo "   plots/      - Generated visualization plots"
echo ""
echo "🔧 USEFUL COMMANDS:"
echo "   bash shell/run_experiment.sh help              # Show help"
echo "   bash shell/visualize_figure3.sh --help        # Visualization options"
echo "   bash shell/run_experiment.sh clean             # Clean up files"
echo "   bash shell/show_structure.sh                   # Show this guide"
echo ""
echo "========================================================================"