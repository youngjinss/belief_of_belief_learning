#!/bin/bash
# Enhanced ToMnet Training Script
# Provides 10x larger datasets and improved generalization

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Enhanced ToMnet Training Pipeline     ${NC}"
echo -e "${BLUE}========================================${NC}"

# Default parameters
EXPERIMENT="figure3"
DEVICE="auto"
BATCH_SIZE=128
LEARNING_RATE=0.001
REGENERATE_DATA=false
TRAIN_INDIVIDUAL=true
TRAIN_MIXED=true
N_WORKERS="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -lr|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --regenerate_data)
            REGENERATE_DATA=true
            shift
            ;;
        --no_individual)
            TRAIN_INDIVIDUAL=false
            shift
            ;;
        --no_mixed)
            TRAIN_MIXED=false
            shift
            ;;
        --only_individual)
            TRAIN_INDIVIDUAL=true
            TRAIN_MIXED=false
            shift
            ;;
        --only_mixed)
            TRAIN_INDIVIDUAL=false
            TRAIN_MIXED=true
            shift
            ;;
        -w|--workers)
            N_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Enhanced ToMnet Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --experiment EXPERIMENT    Experiment type (default: figure3)"
            echo "  -d, --device DEVICE           Device to use (default: auto)"
            echo "  -b, --batch_size SIZE         Batch size (default: 128)"
            echo "  -lr, --learning_rate RATE     Learning rate (default: 0.001)"
            echo "  --regenerate_data             Regenerate datasets even if they exist"
            echo "  --no_individual               Skip individual alpha training"
            echo "  --no_mixed                    Skip mixed/generalization training"
            echo "  --only_individual             Train only individual alpha models"
            echo "  --only_mixed                  Train only mixed/generalization models"
            echo "  -w, --workers N               Number of parallel workers (default: auto)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Enhanced Features:"
            echo "  • 10x larger datasets (1000 agents × 200 episodes)"
            echo "  • Regularization (dropout, weight decay)"
            echo "  • Data augmentation with noise"
            echo "  • Early stopping with patience"
            echo "  • Multiple generalization strategies"
            echo "  • Better optimizers (AdamW, CosineAnnealingWarmRestarts)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Train with all defaults"
            echo "  $0 --regenerate_data --only_mixed    # Regenerate data, train only mixed models"
            echo "  $0 -d cuda:0 -b 256                  # Use specific GPU and larger batch"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Auto-detect device if needed
if [ "$DEVICE" = "auto" ]; then
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        DEVICE="cuda:3"
    elif python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        DEVICE="mps"
    else
        DEVICE="cpu"
    fi
fi

# Auto-detect workers if needed
if [ "$N_WORKERS" = "auto" ]; then
    N_WORKERS=$(python3 -c "import os; print(min(8, os.cpu_count()))")
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Experiment: $EXPERIMENT"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Regenerate data: $REGENERATE_DATA"
echo "  Train individual: $TRAIN_INDIVIDUAL"
echo "  Train mixed: $TRAIN_MIXED"
echo "  Workers: $N_WORKERS"
echo ""

# Check if scripts directory exists
if [ ! -d "scripts" ]; then
    echo -e "${RED}Error: scripts directory not found. Please run from ToMnet_impl root.${NC}"
    exit 1
fi

# Check if train_enhanced.py exists
if [ ! -f "scripts/train_enhanced.py" ]; then
    echo -e "${RED}Error: scripts/train_enhanced.py not found.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data models result

# Prepare command
CMD="python scripts/train_enhanced.py"
CMD="$CMD --experiment $EXPERIMENT"
CMD="$CMD --device $DEVICE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --n_workers $N_WORKERS"

if [ "$REGENERATE_DATA" = true ]; then
    CMD="$CMD --regenerate_data"
fi

if [ "$TRAIN_INDIVIDUAL" = true ]; then
    CMD="$CMD --train_individual"
fi

if [ "$TRAIN_MIXED" = true ]; then
    CMD="$CMD --train_mixed"
fi

echo -e "${BLUE}Starting enhanced training...${NC}"
echo "Command: $CMD"
echo ""

# Run the training
if eval $CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Enhanced Training Completed!          ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Check results in result/$EXPERIMENT/enhanced_training_results.json"
    echo "  2. Run cross-species evaluation:"
    echo "     python scripts/evaluate.py --experiment $EXPERIMENT --use_enhanced_models"
    echo "  3. Compare with original models:"
    echo "     python scripts/compare_models.py --original_results result/$EXPERIMENT/training_results.json --enhanced_results result/$EXPERIMENT/enhanced_training_results.json"
    echo "  4. Visualize improvements:"
    echo "     python scripts/visualize_figure3.py --enhanced_results result/$EXPERIMENT/enhanced_training_results.json"
    echo ""
    
    # Display model summary
    if [ -f "result/$EXPERIMENT/enhanced_training_results.json" ]; then
        echo -e "${BLUE}Trained Models Summary:${NC}"
        python3 -c "
import json
with open('result/$EXPERIMENT/enhanced_training_results.json', 'r') as f:
    results = json.load(f)
if 'enhanced_$EXPERIMENT' in results:
    for model_name, result in results['enhanced_$EXPERIMENT'].items():
        print(f'  • {model_name}:')
        print(f'    - Best val loss: {result.get(\"best_val_loss\", \"N/A\")}')
        if 'final_val_acc' in result:
            print(f'    - Final val acc: {result[\"final_val_acc\"]:.3f}')
        print(f'    - Model: {result.get(\"model_path\", \"N/A\")}')
"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Training Failed!                      ${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the output above for error details."
    exit 1
fi