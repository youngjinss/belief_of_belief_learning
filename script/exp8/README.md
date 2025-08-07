# Experiment 8 - Second-Order Belief (Second Belief) Implementation

This directory implements second-order belief modeling (second belief) in the ToMnet architecture for the belief trading system. The second belief embedding (e_opp2) represents an agent's belief about what others believe about other agents.

## Purpose

Experiment 8 extends the ToMnet architecture to include second-order belief modeling, enabling more sophisticated theory of mind capabilities. The main objectives are:

1. **Second-Order Belief Modeling**: Implement e_opp2 embedding to capture what agents believe others believe
2. **Enhanced ToM Architecture**: Extend the existing 2-stage ToMnet to a 3-stage architecture with cross-attention
3. **Opponent Trajectory Integration**: Use opponent's recent trajectories to inform second belief formation  
4. **Comprehensive Evaluation**: Compare performance with and without second-order belief modeling

## Second Belief Architecture Overview

### Three Core Embeddings

1. **e_char** (Character Embedding) - ✅ Implemented
   - **Input**: Past trajectories from historical episodes
   - **Network**: CharNet  
   - **Output**: 64-dimensional vector encoding behavioral patterns
   - **Purpose**: Captures agent's long-term characteristics and tendencies

2. **e_mental** (Mental State Embedding) - ✅ Implemented
   - **Input**: Recent trajectory and actions
   - **Network**: MentalNet
   - **Output**: Spatial embedding (channels × H × W)
   - **Purpose**: Encodes agent's current mental state from recent behavior

3. **e_opp2** (Second Belief Embedding) - 🚧 **TO BE IMPLEMENTED**
   - **Input**: Concatenation of [e_mental, opponent_recent_trajectory]
   - **Network**: SecondBeliefNet (new component)
   - **Output**: 64-dimensional vector
   - **Purpose**: Models what the agent believes about others' beliefs

### Architecture Evolution

#### Current Architecture (2-stage)
```
Past Episodes → CharNet → e_char ↘  
                                   → PredNet → Predictions
Recent Trajectory → MentalNet → e_mental ↗
```

#### Target Architecture (3-stage with Second Belief)
```
Past Episodes → CharNet → e_char ↘
                                   ↘
Recent Trajectory → MentalNet → e_mental → [concat with opponent_recent_trajectory] → SecondBeliefNet → e_opp2
                                           ↗                                                              ↓
                                          PredNet ← Cross-Attention ← [e_char, e_mental, e_opp2] ←────┘
```

## Implementation Plan

### Phase 1: Data Preparation
- [ ] Modify `_create_trajectory_tensor` to support perspective switching
- [ ] Add `opponent_recent_trajectory` extraction in sample creation functions
- [ ] Update data loader to include opponent trajectory fields
- [ ] Verify data generation with unit tests

### Phase 2: Model Architecture  
- [ ] Implement `SecondBeliefNet` class
- [ ] Implement `CrossAttentionModule` class
- [ ] Modify `PredNet` to accept three embeddings via cross-attention
- [ ] Update `ToMnet` class with new components and `use_second_belief` flag

### Phase 3: Training Integration
- [ ] Update training loop to pass opponent trajectories
- [ ] Ensure backward compatibility (optional opponent data)
- [ ] Update model checkpointing for new parameters
- [ ] Add second belief evaluation metrics

### Phase 4: Evaluation & Visualization
- [ ] Implement e_opp2 visualization functions
- [ ] Create attention weight visualizations  
- [ ] Update existing plots to include second belief analysis
- [ ] Compare performance with/without second-order belief modeling

## Key Implementation Components

### SecondBeliefNet Architecture
The core component for second-order belief modeling:

```python
class SecondBeliefNet(nn.Module):
    def __init__(self, n_ement, n_eopp2, channels_in, hidden_size=128):
        # 1. Trajectory encoder (Conv + ResNet blocks)
        # 2. Mental state processor (handles spatial e_mental)  
        # 3. Fusion layer (combines trajectory and mental state)
        # 4. LSTM for temporal processing
        # 5. Output projection to n_eopp2
```

### Cross-Attention Mechanism
Combines all three embeddings for enhanced prediction:

```python
class CrossAttentionModule(nn.Module):
    def __init__(self, n_echar, n_ement, n_eopp2, hidden_dim):
        # Multi-head attention between embeddings
        # Query: current state features
        # Keys/Values: [e_char, e_mental, e_opp2]
```

### Configuration Updates
```python
model_config = {
    "n_eopp2": 64,  # Second belief embedding dimension
    "use_second_belief": True,  # Enable second belief modeling
    "second_belief_hidden": 128,  # Hidden size for SecondBeliefNet
    "attention_heads": 8,  # Number of attention heads
    "attention_hidden": 256,  # Hidden dimension for attention
}
```

## Data Structure Enhancements

### Opponent Trajectory Integration
The implementation requires extracting opponent trajectories with perspective switching:

```python
def create_achiever_sample(self, parsed_data):
    # ... existing code ...
    # Add opponent's recent trajectory  
    opponent_trajectory = self._create_trajectory_tensor(
        parsed_data["maze"],
        parsed_data["trajectory_steps"], 
        "blocker",  # Opponent perspective
        parsed_data["trajectory_length"]
    )
    return {
        # ... existing fields ...
        "opponent_recent_trajectory": opponent_trajectory,
        "opponent_actions": blocker_actions,
    }
```

### Trajectory File Structure
Current trajectory files contain both agents' data, enabling opponent extraction:
```
[achiever_x, achiever_y][blocker_x, blocker_y] : achiever_action,blocker_action : achiever_interaction,blocker_interaction
```

For opponent trajectory extraction:
1. **Achiever samples**: Use blocker positions/actions as opponent data
2. **Blocker samples**: Use achiever positions/actions as opponent data

### Backward Compatibility
- Make `opponent_recent_trajectory` optional in data loading
- Add `use_second_belief` configuration flag (default: False for existing models)
- Skip SecondBeliefNet computation when flag is disabled
- Maintain existing PredNet path for backward compatibility

## Expected Performance Improvements

### Theory of Mind Enhancement
With second-order belief modeling, we expect:

1. **Better Opponent Prediction**: Agents can model what opponents believe, leading to more accurate predictions of opponent behavior
2. **Strategic Awareness**: Enhanced understanding of multi-step strategic interactions  
3. **Improved Competition**: Better performance in competitive multi-agent scenarios
4. **Richer Mental Models**: More sophisticated internal representations of other agents

### Evaluation Metrics
- **Prediction Accuracy**: Compare action prediction accuracy with/without e_opp2
- **Attention Analysis**: Visualize which embeddings contribute most to predictions
- **Second Belief Accuracy**: Track how well e_opp2 predicts opponent behavior
- **Computational Overhead**: Measure additional computational cost of second belief processing

## Technical Considerations

### Memory Usage
- Adding opponent trajectories doubles trajectory data storage
- Consider lazy loading and shared storage with perspective indexing
- Implement efficient data structures for opponent trajectory access

### Single-Agent Mode Handling
- Set `opponent_recent_trajectory` to None in single-agent episodes
- SecondBeliefNet should gracefully handle None inputs
- Use zero vector or learned "no-opponent" embedding when opponent absent

### Computational Efficiency  
- Cross-attention mechanism adds computational overhead
- Profile and optimize attention bottlenecks
- Consider efficient attention implementations (e.g., sparse attention)
- Monitor training time and memory usage increases

### Interpretability
- Implement visualization tools for attention weights
- Add debugging capabilities to inspect e_opp2 embeddings
- Create comparative analysis tools (with/without second belief)

## Usage 

### Running Second Belief Training
```bash
# Enable second belief in config.py
config.model_config["use_second_belief"] = True

# Run training with second belief
python script/exp8/train.py

# Run complete pipeline
bash shell/exp8/run_exp8.sh all
```

### Configuration for Second Belief
```python
# In config.py
model_config = {
    # ... existing config ...
    "n_eopp2": 64,  # Second belief embedding dimension
    "use_second_belief": True,  # Enable second belief modeling
    "second_belief_hidden": 128,  # Hidden size for SecondBeliefNet
    "attention_heads": 8,  # Number of attention heads
    "attention_hidden": 256,  # Hidden dimension for attention
}
```

### Comparative Evaluation
```bash
# Train baseline model (without second belief)
python script/exp8/train.py --config_override --use_second_belief false

# Train enhanced model (with second belief)  
python script/exp8/train.py --config_override --use_second_belief true

# Compare results
python script/exp8/evaluate.py --compare_models baseline enhanced
```

## Testing Strategy

### Unit Tests
1. **Perspective Switching**: Test trajectory creation from opponent's perspective
2. **SecondBeliefNet Architecture**: Test network with various input sizes and configurations
3. **Cross-Attention Mechanism**: Validate attention computation and gradient flow
4. **Backward Compatibility**: Ensure existing models work without second belief

### Integration Tests
1. **Full Forward Pass**: Test complete model pipeline with second belief enabled
2. **Training Loop**: Verify training step with enhanced architecture
3. **Data Loading**: Test opponent trajectory extraction and loading
4. **Evaluation Pipeline**: Test model evaluation with/without second belief

### Performance Tests
1. **Prediction Accuracy**: Compare accuracy improvements with second belief
2. **Computational Overhead**: Measure training/inference time increases
3. **Memory Usage**: Monitor memory consumption of enhanced architecture
4. **Attention Analysis**: Validate attention patterns and interpretability

## Visualization Enhancements

### Second Belief Embeddings
- **3D Embedding Space**: Visualize e_char, e_mental, e_opp2 relationships
- **Cluster Analysis**: Group embeddings by agent type and strategic context
- **Embedding Evolution**: Track how e_opp2 changes during training

### Attention Analysis
- **Attention Weight Heatmaps**: Show which embeddings influence predictions
- **Cross-Attention Patterns**: Visualize attention flow between embedding types
- **Strategic Context Analysis**: Compare attention patterns in different game scenarios

### Performance Metrics
- **Comparative Accuracy**: Track prediction improvements with second belief
- **Computational Profiling**: Monitor training time and memory usage
- **Ablation Studies**: Analyze contribution of individual components

## Version History

- **v1.0** (Initial): Based on exp6 unified single/multi-agent framework
- **v2.0** (Planned): Second-order belief implementation
  - SecondBeliefNet architecture
  - Cross-attention mechanism
  - Opponent trajectory integration
  - Enhanced evaluation and visualization
- **v2.1** (2025-08-04): Major refactoring and bug fixes
  - Moved `spatialize_action` function to `utils.py` to prevent code duplication
  - Renamed variables for clarity: `trajectory→self_states`, `opponent_trajectory→oppo_states`, `actions→self_actions`, `opponent_actions→oppo_actions`
  - Removed `extract_actions_from_trajectory` function from all networks
  - Fixed critical bug where models were using zero-filled actions instead of real action data
  - Networks now properly concatenate states with spatialized actions using `spatialize_action` from utils
  - Updated all function signatures and calls across exp8 scripts
  - Fixed tensor shape mismatches in MentalNet and SecondBeliefNet
  - Comprehensive testing validates all ToMnet configurations work correctly

## Research Applications

### Theory of Mind Studies
- Compare performance across different levels of theory of mind
- Analyze attention patterns to understand belief formation
- Study emergence of strategic reasoning in competitive environments

### Multi-Agent Learning
- Investigate how second-order beliefs affect learning dynamics
- Compare convergence properties with/without second belief
- Analyze stability of multi-agent training with enhanced ToM  

### Cognitive Science Insights
- Model human-like theory of mind development
- Study computational requirements of second-order belief modeling
- Investigate scalability to higher-order beliefs (third-order, etc.)

## V2 Environment Integration and Optimization

### Environment Version Compatibility Analysis

During exp8 integration, a comprehensive analysis was performed to ensure V2 environments (`achiever_blocker_v2.py`, `keydoor_v2.py`) are logically identical to V1 versions except for partial observation capabilities.

#### Issues Identified and Fixed

**1. Door Placement Algorithm Inconsistency**
- **Problem**: V2 environments used different door placement logic than V1
  - V1: Places doors only at 4 fixed wall center positions
  - V2: Places doors at any available wall position (much more randomization)
- **Impact**: Different game difficulty and training data distribution
- **Solution**: ✅ Updated both V2 environments to use V1's center-only placement algorithm

**2. Method Signature Inconsistency (KeyDoor only)**
- **Problem**: KeyDoor V1 vs V2 had incompatible method signatures
  - V1: `_auto_pickup_key()`, `_auto_open_door()` - use `self.agent_pos`
  - V2: `_auto_pickup_key(agent_pos)`, `_auto_open_door(agent_pos)` - take position parameter
- **Impact**: Environments not drop-in compatible
- **Solution**: ✅ Updated KeyDoor V2 to match V1 signatures (AchieverBlocker was already consistent)

**3. Performance Optimization Opportunities**
- **Problem**: KeyDoor V2 used inefficient grid scanning for position lookups
- **Solution**: ✅ Added position storage (`self.door_positions`, `self.key_positions`) like AchieverBlocker

#### Implemented Optimizations

**1. Shared Base Class Architecture**
```python
# Created BaseEnvV2 for code reuse
class BaseEnvV2(MiniGridEnv):
    def _get_observations(self):
        if self.observability == "partial":
            return self._get_partial_observations()
        else:
            # Skip visibility computation in full mode
            return self._get_full_observations()
```

**2. Position Storage Optimization**
```python
# Added to KeyDoor V2 (AchieverBlocker already had this)
self.door_positions = {}  # color -> (x, y) position
self.key_positions = {}   # color -> (x, y) position (None if consumed)

def _get_key_positions(self):
    # Efficient lookup instead of grid scanning
    return {color: pos for color, pos in self.key_positions.items() if pos is not None}
```

**3. Optimized Observation Routing**
- Full observation mode skips unnecessary visibility computation
- Partial observation mode uses shared filtering logic from base class

#### Final V2 Environment State

✅ **Logical Compatibility**: V1 and V2 are now identical except for partial observation  
✅ **Performance**: V2 includes optimized position storage and observation routing  
✅ **Code Organization**: Shared BaseEnvV2 eliminates duplication  
✅ **Drop-in Replacement**: V2 can replace V1 with added partial observation capability  

The V2 environments are now ready for integration into exp8 scripts with full backward compatibility and enhanced partial observation support.

## Acknowledgments

Built upon the exp6 unified single/multi-agent framework, extending it to include second-order belief modeling for enhanced theory of mind capabilities in competitive multi-agent environments.