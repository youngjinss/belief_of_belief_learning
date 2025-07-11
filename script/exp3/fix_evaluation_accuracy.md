# Evaluation Accuracy Fix Summary

## Problem
Training validation accuracy: 52.8%
Evaluation accuracy: 32.4%

## Root Causes Identified

1. **Past Episodes Generation Mismatch**
   - Training uses `goal_ranks` for past episode generation
   - Evaluation was using `goals` instead
   - This causes mismatched past episodes between training and evaluation

2. **Missing Data in Test Dataset**
   - Test dataset only included 3 tensors (trajectories, actions, goals)
   - Training uses 7 tensors including goal_ranks, consumption_labels, sr_labels
   - This prevented proper past episode generation

3. **Fixed vs Dynamic Timesteps**
   - Training uses dynamic trajectory slicing based on actual trajectory length
   - Some evaluation functions used fixed timesteps (e.g., `current_timestep = data_config["time_step"]`)
   - This causes incorrect current state extraction

4. **Hardcoded Channel Configuration**
   - Evaluation hardcoded `current_state_channels = 8`
   - Should use configuration value like training does

5. **Missing rank_threshold Parameter**
   - Past episode generation in evaluation didn't pass `rank_threshold`
   - This parameter controls goal matching logic

## Fixes Applied

### 1. Updated Test Dataset Creation
```python
# Before
test_dataset = TensorDataset(
    test_data["trajectories"], test_data["actions"], test_data["goals"]
)

# After
test_dataset = TensorDataset(
    test_data["trajectories"],
    test_data["actions"],
    test_data["goals"],
    test_data["goal_ranks"],
    test_data["goal_rewards"],
    test_data["consumption_labels"],
    test_data["sr_labels"],
)
```

### 2. Fixed Past Episode Generation
```python
# Before
past_episodes = generate_past_episodes_from_batch(
    trajectories, goals, batch_size, ...
)

# After
past_episodes = generate_past_episodes_from_batch(
    trajectories, goal_ranks, batch_size, ...,
    rank_threshold=data_config.get("rank_threshold", 1)
)
```

### 3. Applied Dynamic Trajectory Slicing
```python
# Find effective lengths dynamically (matching training)
traj_sums = trajectories.sum(dim=(2, 3, 4))
non_zero_mask = traj_sums > 0
# ... (full vectorized logic)
effective_lengths = masked_indices.max(dim=1)[0].clamp(min=0)

# Extract current state at correct timestep
batch_indices = torch.arange(batch_size, device=trajectories.device)
current_state = trajectories[batch_indices, effective_lengths, :current_state_channels]
```

### 4. Used Configuration for Channel Settings
```python
# Before
current_state_channels = 8  # Hardcoded

# After
current_state_channels = data_config.get("current_state_channels", 8)
recent_trajectory = trajectories[:, :, :current_state_channels]
```

## Testing the Fix

To verify the fixes work correctly:

1. Run evaluation with the updated code:
```bash
python script/exp3/evaluate.py --config_override --test_data_dir data/exp3/test --model_path results/exp3/best_model.pth --result_dir results/exp3
```

2. Compare the new evaluation accuracy with training validation accuracy
3. The accuracies should now be much closer (within 1-2%)

## Additional Recommendations

1. **Add Validation During Training**: Log the exact evaluation logic during training validation to ensure consistency

2. **Unit Tests**: Create unit tests that verify training and evaluation use identical data processing

3. **Configuration Validation**: Add checks to ensure all required configuration parameters are loaded correctly

4. **Logging**: Add detailed logging of data shapes and configuration values during both training and evaluation