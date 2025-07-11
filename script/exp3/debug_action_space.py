#!/usr/bin/env python3
"""
Debug the action space and data processing issues
"""
import json
import numpy as np


def debug_action_space():
    """Debug action space and data distribution issues"""
    
    print("="*60)
    print("DEBUGGING ACTION SPACE AND DATA ISSUES")
    print("="*60)
    
    # 1. Check the confusion matrix dimensions
    with open("results/evaluation_results_exp3.json", "r") as f:
        results = json.load(f)
    
    cm = np.array(results["confusion_matrix"])
    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Expected shape for KeyDoor: (7, 7)")
    print(f"Actual shape: {cm.shape}")
    
    if cm.shape != (7, 7):
        print("🚨 PROBLEM: Wrong confusion matrix size!")
        print("This suggests:")
        print("  - Model is predicting wrong number of actions")
        print("  - Action labels are being mapped incorrectly")
        print("  - Test data has different action space than training")
    
    print(f"\nConfusion matrix:")
    print(cm)
    
    # 2. Analyze action distributions
    print(f"\n" + "="*40)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*40)
    
    # Calculate true and predicted action distributions
    true_actions = cm.sum(axis=1)  # Sum each row (true labels)
    pred_actions = cm.sum(axis=0)  # Sum each column (predictions)
    
    print(f"True action distribution (test data):")
    for i, count in enumerate(true_actions):
        if count > 0:
            pct = (count / true_actions.sum()) * 100
            print(f"  Action {i}: {count} samples ({pct:.1f}%)")
    
    print(f"\nPredicted action distribution (model output):")
    for i, count in enumerate(pred_actions):
        if count > 0:
            pct = (count / pred_actions.sum()) * 100
            print(f"  Action {i}: {count} predictions ({pct:.1f}%)")
    
    # 3. Check per-action accuracy from results
    print(f"\n" + "="*40)
    print("PER-ACTION ACCURACY")
    print("="*40)
    
    action_accuracies = results["action_accuracy"]
    print("Action accuracies from evaluation:")
    for action, acc in action_accuracies.items():
        print(f"  {action}: {acc:.4f}")
    
    print(f"\nTotal actions found in evaluation: {len(action_accuracies)}")
    print(f"Expected for KeyDoor: 7 actions")
    
    if len(action_accuracies) < 7:
        print("🚨 PROBLEM: Missing actions in evaluation!")
        missing_actions = []
        for i in range(7):
            if f"action_{i}" not in action_accuracies:
                missing_actions.append(i)
        print(f"Missing actions: {missing_actions}")
    
    # 4. Check model bias
    print(f"\n" + "="*40)
    print("MODEL BIAS ANALYSIS")
    print("="*40)
    
    # Calculate which actions the model is biased toward
    total_predictions = pred_actions.sum()
    most_predicted_action = np.argmax(pred_actions)
    most_predicted_count = pred_actions[most_predicted_action]
    bias_pct = (most_predicted_count / total_predictions) * 100
    
    print(f"Most predicted action: {most_predicted_action}")
    print(f"Predictions for action {most_predicted_action}: {most_predicted_count}/{total_predictions} ({bias_pct:.1f}%)")
    
    if bias_pct > 50:
        print(f"🚨 SEVERE MODEL BIAS: Model predicts action {most_predicted_action} {bias_pct:.1f}% of the time!")
        print("This suggests:")
        print("  - Model collapsed to predicting one action")
        print("  - Training data was imbalanced")
        print("  - Evaluation labels are wrong")
    
    # 5. Compare with training accuracy
    print(f"\n" + "="*40)
    print("TRAINING VS EVALUATION COMPARISON")
    print("="*40)
    
    with open("results/training_history.json", "r") as f:
        training = json.load(f)
    
    best_val_acc = max(training["val_action_accuracy"])
    eval_acc = results["accuracy"]
    drop = best_val_acc - eval_acc
    
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"Evaluation accuracy: {eval_acc:.4f} ({eval_acc*100:.1f}%)")
    print(f"Accuracy drop: {drop:.4f} ({(drop/best_val_acc)*100:.1f}%)")
    
    # 6. Check N_past evaluation consistency
    print(f"\n" + "="*40)
    print("N_PAST EVALUATION CONSISTENCY")
    print("="*40)
    
    with open("results/n_past_evaluation_results.json", "r") as f:
        n_past = json.load(f)
    
    n_past_accs = [metrics["accuracy"] for metrics in n_past.values()]
    print(f"N_past accuracies: {n_past_accs}")
    
    if len(set(n_past_accs)) == 1:
        print("🚨 PROBLEM: All N_past evaluations give identical results!")
        print("This suggests:")
        print("  - Past episode generation is not working")
        print("  - Model is ignoring past episodes")
        print("  - Evaluation pipeline has a bug")
    
    # 7. Provide specific fixes
    print(f"\n" + "="*60)
    print("SPECIFIC DEBUGGING STEPS")
    print("="*60)
    
    print("1. CHECK ACTION SPACE:")
    print("   - Verify model.action_space == 7")
    print("   - Check if model output has 7 dimensions")
    print("   - Verify action labels in test data are 0-6")
    
    print("\n2. CHECK DATA PREPROCESSING:")
    print("   - Verify test data uses same action encoding as training")
    print("   - Check if action targets are correctly extracted")
    print("   - Verify trajectory slicing produces correct labels")
    
    print("\n3. CHECK MODEL CONFIGURATION:")
    print("   - Verify model_config.json has action_space=7")
    print("   - Check if model architecture matches training")
    print("   - Verify model.load_state_dict() works correctly")
    
    print("\n4. CHECK EVALUATION LOOP:")
    print("   - Add debug prints in evaluate_model function")
    print("   - Check action_targets values and ranges")
    print("   - Verify model forward pass output dimensions")
    
    print(f"\n🎯 MOST LIKELY ISSUE:")
    if cm.shape[0] == 4:
        print("The model or evaluation is treating this as a 4-action problem instead of 7-action!")
        print("Check:")
        print("  - Model configuration (action_space parameter)")
        print("  - Action label encoding in test data")
        print("  - Goal vs action confusion (goals=4, actions=7)")


if __name__ == "__main__":
    debug_action_space()