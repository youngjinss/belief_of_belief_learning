#!/usr/bin/env python3
"""
Analyze existing evaluation results to understand the accuracy issue
"""
import json
import numpy as np


def analyze_evaluation_results():
    """Analyze the evaluation results to understand the accuracy discrepancy"""
    
    print("="*60)
    print("ANALYZING EVALUATION RESULTS")
    print("="*60)
    
    # Load training history
    try:
        with open("results/training_history.json", "r") as f:
            training_history = json.load(f)
        print("✓ Training history loaded")
    except Exception as e:
        print(f"✗ Failed to load training history: {e}")
        return
    
    # Load evaluation results
    try:
        with open("results/evaluation_results_exp3.json", "r") as f:
            evaluation_results = json.load(f)
        print("✓ Evaluation results loaded")
    except Exception as e:
        print(f"✗ Failed to load evaluation results: {e}")
        return
    
    # Load N_past evaluation results
    try:
        with open("results/n_past_evaluation_results.json", "r") as f:
            n_past_results = json.load(f)
        print("✓ N_past evaluation results loaded")
    except Exception as e:
        print(f"✗ Failed to load N_past evaluation results: {e}")
        n_past_results = None
    
    print("\n" + "="*60)
    print("TRAINING PERFORMANCE")
    print("="*60)
    
    # Analyze training history
    final_epoch = len(training_history["epoch"])
    final_train_action_acc = training_history["train_action_accuracy"][-1]
    final_val_action_acc = training_history["val_action_accuracy"][-1]
    final_train_goal_acc = training_history["train_goal_accuracy"][-1]
    final_val_goal_acc = training_history["val_goal_accuracy"][-1]
    final_train_loss = training_history["train_loss"][-1]
    final_val_loss = training_history["val_loss"][-1]
    
    print(f"Final epoch: {final_epoch}")
    print(f"Final training action accuracy: {final_train_action_acc:.4f} ({final_train_action_acc*100:.1f}%)")
    print(f"Final validation action accuracy: {final_val_action_acc:.4f} ({final_val_action_acc*100:.1f}%)")
    print(f"Final training goal accuracy: {final_train_goal_acc:.4f} ({final_train_goal_acc*100:.1f}%)")
    print(f"Final validation goal accuracy: {final_val_goal_acc:.4f} ({final_val_goal_acc*100:.1f}%)")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Find best validation accuracies
    best_val_action_acc = max(training_history["val_action_accuracy"])
    best_val_goal_acc = max(training_history["val_goal_accuracy"])
    best_val_loss = min(training_history["val_loss"])
    
    print(f"\nBest validation action accuracy: {best_val_action_acc:.4f} ({best_val_action_acc*100:.1f}%)")
    print(f"Best validation goal accuracy: {best_val_goal_acc:.4f} ({best_val_goal_acc*100:.1f}%)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    print("\n" + "="*60)
    print("EVALUATION PERFORMANCE")
    print("="*60)
    
    # Analyze evaluation results
    eval_accuracy = evaluation_results["accuracy"]
    eval_f1 = evaluation_results["f1_score"]
    eval_precision = evaluation_results["precision"]
    eval_recall = evaluation_results["recall"]
    eval_samples = evaluation_results["n_samples"]
    
    print(f"Evaluation accuracy: {eval_accuracy:.4f} ({eval_accuracy*100:.1f}%)")
    print(f"Evaluation F1 score: {eval_f1:.4f}")
    print(f"Evaluation precision: {eval_precision:.4f}")
    print(f"Evaluation recall: {eval_recall:.4f}")
    print(f"Number of samples: {eval_samples}")
    
    # Per-action accuracy
    print(f"\nPer-action accuracy:")
    for action, acc in evaluation_results["action_accuracy"].items():
        print(f"  {action}: {acc:.4f}")
    
    # Confidence statistics
    conf_stats = evaluation_results["confidence_stats"]
    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {conf_stats['mean_confidence']:.4f}")
    print(f"  Std confidence: {conf_stats['std_confidence']:.4f}")
    print(f"  Min confidence: {conf_stats['min_confidence']:.4f}")
    print(f"  Max confidence: {conf_stats['max_confidence']:.4f}")
    
    print("\n" + "="*60)
    print("ACCURACY DISCREPANCY ANALYSIS")
    print("="*60)
    
    # Calculate discrepancy
    val_to_eval_drop = best_val_action_acc - eval_accuracy
    val_to_eval_drop_pct = (val_to_eval_drop / best_val_action_acc) * 100
    
    print(f"Best validation accuracy: {best_val_action_acc*100:.1f}%")
    print(f"Evaluation accuracy: {eval_accuracy*100:.1f}%")
    print(f"Accuracy drop: {val_to_eval_drop:.4f} ({val_to_eval_drop_pct:.1f}%)")
    
    if val_to_eval_drop > 0.15:  # More than 15% drop
        print(f"\n🚨 LARGE ACCURACY DROP DETECTED!")
        print(f"This suggests:")
        print(f"  1. Data preprocessing differences between training and evaluation")
        print(f"  2. Model architecture mismatch during loading")
        print(f"  3. Different data distributions (train vs test)")
        print(f"  4. Bugs in evaluation pipeline")
    elif val_to_eval_drop > 0.05:  # 5-15% drop
        print(f"\n⚠️  MODERATE ACCURACY DROP")
        print(f"This is somewhat expected but could be improved")
    else:
        print(f"\n✅ GOOD: Small accuracy drop, within expected range")
    
    # Analyze N_past results if available
    if n_past_results:
        print("\n" + "="*60)
        print("N_PAST EVALUATION ANALYSIS")
        print("="*60)
        
        n_past_accuracies = {}
        for n_str, metrics in n_past_results.items():
            n = int(n_str)
            acc = metrics["accuracy"]
            n_past_accuracies[n] = acc
            print(f"N_past={n}: {acc:.4f} ({acc*100:.1f}%)")
        
        # Check if accuracy varies significantly with N_past
        accuracies = list(n_past_accuracies.values())
        if accuracies:
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            acc_range = max_acc - min_acc
            
            print(f"\nN_past accuracy range: {min_acc:.4f} to {max_acc:.4f} (range: {acc_range:.4f})")
            
            if acc_range > 0.1:
                print("🔍 Large variation in N_past accuracy suggests past episode generation issues")
            else:
                print("✅ Consistent accuracy across N_past values")
    
    # Analyze confusion matrix
    if "confusion_matrix" in evaluation_results:
        cm = np.array(evaluation_results["confusion_matrix"])
        print(f"\n" + "="*60)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*60)
        
        print("Confusion matrix shape:", cm.shape)
        
        # Calculate per-class precision and recall
        if cm.size > 0:
            n_classes = cm.shape[0]
            print(f"Number of classes: {n_classes}")
            
            for i in range(min(n_classes, 7)):  # KeyDoor has 7 actions
                true_positives = cm[i, i] if i < cm.shape[1] else 0
                predicted_positives = cm[:, i].sum() if i < cm.shape[1] else 0
                actual_positives = cm[i, :].sum() if i < cm.shape[0] else 0
                
                precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                recall = true_positives / actual_positives if actual_positives > 0 else 0
                
                print(f"Action {i}: Precision={precision:.3f}, Recall={recall:.3f}, Support={actual_positives}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if val_to_eval_drop > 0.15:
        print("🔧 FIXES NEEDED:")
        print("1. Check data preprocessing consistency (trajectory slicing, past episodes)")
        print("2. Verify model loading uses correct configuration")
        print("3. Ensure evaluation uses same data format as training")
        print("4. Check if test data distribution matches training data")
    elif val_to_eval_drop > 0.05:
        print("🔧 POSSIBLE IMPROVEMENTS:")
        print("1. Fine-tune evaluation parameters")
        print("2. Check data augmentation consistency")
        print("3. Verify batch processing logic")
    else:
        print("✅ EVALUATION LOOKS GOOD")
        print("The accuracy drop is within expected range")
    
    print(f"\nFinal assessment:")
    print(f"Training reached {best_val_action_acc*100:.1f}% validation accuracy")
    print(f"Evaluation achieved {eval_accuracy*100:.1f}% accuracy")
    print(f"Drop of {val_to_eval_drop_pct:.1f}% {'is concerning' if val_to_eval_drop_pct > 20 else 'is acceptable'}")


if __name__ == "__main__":
    analyze_evaluation_results()