#!/bin/bash
# Test script to validate ToMnet action likelihood against paper results
# Expected results from paper:
# - N_past=5, trained alpha=0.01: action likelihood ~1.0
# - N_past=1, trained alpha=0.01: action likelihood ~1.0  
# - N_past=0, trained alpha=0.01: action likelihood ~0.2

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Test configuration
TEST_ALPHA=0.01
N_PAST_VALUES=(0 1 5)
EXPECTED_LIKELIHOODS=(0.2 1.0 1.0)  # Expected values from paper
TOLERANCE=0.1  # Tolerance for matching expected values

# Create test directories
TEST_DIR="test_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
LOG_FILE="$TEST_DIR/test_experiment.log"

print_info "Starting ToMnet action likelihood validation test"
print_info "Test results will be saved to: $TEST_DIR"
print_info "Log file: $LOG_FILE"

# Function to check if value is within expected range
check_likelihood() {
    local n_past=$1
    local actual=$2
    local expected=$3
    local tolerance=$4
    
    # Use awk for floating point comparison
    local diff=$(awk "BEGIN {print ($actual - $expected < 0) ? ($expected - $actual) : ($actual - $expected)}")
    local within_tolerance=$(awk "BEGIN {print ($diff <= $tolerance) ? 1 : 0}")
    
    if [ "$within_tolerance" -eq 1 ]; then
        print_success "N_past=$n_past: PASS (Expected: $expected, Actual: $actual, Diff: $diff)"
        return 0
    else
        print_error "N_past=$n_past: FAIL (Expected: $expected, Actual: $actual, Diff: $diff)"
        return 1
    fi
}

# Start logging
{
    echo "ToMnet Action Likelihood Test - $(date)"
    echo "============================================"
    echo "Testing conditions:"
    echo "- Trained alpha: $TEST_ALPHA"
    echo "- N_past values: ${N_PAST_VALUES[*]}"
    echo "- Expected likelihoods: ${EXPECTED_LIKELIHOODS[*]}"
    echo ""
} >> "$LOG_FILE"

print_info "Step 1: Training model with alpha=$TEST_ALPHA"

# Train model for alpha=0.01 only
print_info "Training ToMnet model for alpha=$TEST_ALPHA..."
python scripts/train.py \
    --experiment figure3 \
    --n_agents 100 \
    --n_epochs 20 \
    --batch_size 32 \
    --alpha_values "$TEST_ALPHA" >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    print_error "Training failed! Check log: $LOG_FILE"
    exit 1
fi
print_success "Training completed!"

print_info "Step 2: Creating evaluation configuration files"

# Create model paths JSON for evaluation
MODEL_PATHS_JSON="$TEST_DIR/model_paths.json"
cat > "$MODEL_PATHS_JSON" << EOF
{
    "alpha_${TEST_ALPHA}": "models/figure3_${TEST_ALPHA}_best.pth"
}
EOF

# Create dataset paths JSON for evaluation  
DATA_PATHS_JSON="$TEST_DIR/data_paths.json"
cat > "$DATA_PATHS_JSON" << EOF
{
    "alpha_${TEST_ALPHA}": "data/figure3_alpha_${TEST_ALPHA}.pkl"
}
EOF

print_info "Step 3: Running evaluation for different N_past values"

# Run evaluation
RESULTS_FILE="$TEST_DIR/evaluation_results.pkl"
python scripts/evaluate.py \
    --experiment figure3 \
    --model_paths_json "$MODEL_PATHS_JSON" \
    --data_paths_json "$DATA_PATHS_JSON" \
    --output_path "$RESULTS_FILE" >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    print_error "Evaluation failed! Check log: $LOG_FILE"
    exit 1
fi
print_success "Evaluation completed!"

print_info "Step 4: Extracting and validating action likelihood results"

# Create Python script to extract action likelihoods
EXTRACT_SCRIPT="$TEST_DIR/extract_likelihoods.py"
cat > "$EXTRACT_SCRIPT" << 'EOF'
import pickle
import sys
import numpy as np

def extract_action_likelihoods(results_file):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    if 'figure3a' not in results:
        print("ERROR: figure3a results not found in evaluation results")
        return {}
    
    figure3a = results['figure3a']
    n_past_values = figure3a.get('n_past_values', [])
    action_likelihoods_by_n_past = figure3a.get('action_likelihoods_by_n_past', {})
    
    extracted = {}
    for n_past in n_past_values:
        if n_past in action_likelihoods_by_n_past:
            likelihoods = action_likelihoods_by_n_past[n_past]
            if likelihoods:
                mean_likelihood = np.mean(likelihoods)
                extracted[n_past] = mean_likelihood
                print(f"N_past={n_past}: Action likelihood = {mean_likelihood:.4f}")
            else:
                print(f"N_past={n_past}: No likelihood data found")
        else:
            print(f"N_past={n_past}: Not found in results")
    
    return extracted

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_likelihoods.py <results_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    likelihoods = extract_action_likelihoods(results_file)
    
    # Save extracted results
    output_file = results_file.replace('.pkl', '_likelihoods.txt')
    with open(output_file, 'w') as f:
        for n_past, likelihood in likelihoods.items():
            f.write(f"{n_past} {likelihood:.6f}\n")
    
    print(f"Results saved to: {output_file}")
EOF

# Extract action likelihoods
print_info "Extracting action likelihood values..."
LIKELIHOOD_FILE="$TEST_DIR/evaluation_results_likelihoods.txt"
python "$EXTRACT_SCRIPT" "$RESULTS_FILE" >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    print_error "Failed to extract action likelihoods! Check log: $LOG_FILE"
    exit 1
fi

print_info "Step 5: Validating results against paper expectations"

# Read extracted likelihoods and validate
if [ ! -f "$LIKELIHOOD_FILE" ]; then
    print_error "Likelihood results file not found: $LIKELIHOOD_FILE"
    exit 1
fi

print_info "Comparing results with paper expectations:"
echo "Expected vs Actual Action Likelihoods:"
echo "======================================"

VALIDATION_PASSED=true

# Read and validate each N_past result
while IFS=' ' read -r n_past actual_likelihood; do
    # Find expected value for this N_past
    expected_likelihood=""
    for i in "${!N_PAST_VALUES[@]}"; do
        if [ "${N_PAST_VALUES[$i]}" -eq "$n_past" ]; then
            expected_likelihood="${EXPECTED_LIKELIHOODS[$i]}"
            break
        fi
    done
    
    if [ -n "$expected_likelihood" ]; then
        if ! check_likelihood "$n_past" "$actual_likelihood" "$expected_likelihood" "$TOLERANCE"; then
            VALIDATION_PASSED=false
        fi
    else
        print_warning "No expected value found for N_past=$n_past"
    fi
    
done < "$LIKELIHOOD_FILE"

echo ""

# Final result
if [ "$VALIDATION_PASSED" = true ]; then
    print_success "🎉 All tests PASSED! Action likelihoods match paper expectations."
    echo "RESULT: VALIDATION SUCCESSFUL" >> "$LOG_FILE"
    exit 0
else
    print_error "❌ Some tests FAILED! Action likelihoods do not match paper expectations."
    print_warning "This suggests there may be an issue with the ToMnet implementation."
    print_info "Please check the following:"
    print_info "1. Data generation and preprocessing"
    print_info "2. Model architecture (CharacterNet, PredictionNet)"
    print_info "3. Training hyperparameters and convergence"
    print_info "4. Evaluation methodology (action likelihood calculation)"
    echo "RESULT: VALIDATION FAILED" >> "$LOG_FILE"
    
    print_info "For debugging, check the detailed logs at: $LOG_FILE"
    print_info "Raw results available at: $RESULTS_FILE"
    print_info "Extracted likelihoods at: $LIKELIHOOD_FILE"
    
    exit 1
fi