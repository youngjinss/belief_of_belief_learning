import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def analyze_beliefs(model, dataloader, device="cpu"):
    """
    Analyze the belief states learned by the model.

    Args:
        model: The trained TFT model
        dataloader: DataLoader for the data to analyze
        device: Device to run the model on

    Returns:
        dict: Dictionary containing belief analysis results
    """
    model.eval()
    belief_states = []
    timestamps = []
    actions = []

    with torch.no_grad():
        for batch in dataloader:
            ohlcv = batch["ohlcv"].to(device)
            self_actions = batch["self_actions"].to(device)
            other_actions = batch["other_actions"].to(device)

            # Forward pass to get predictions and attention weights
            _, attention_weights = model(ohlcv, self_actions, other_actions)

            # Store the results for analysis
            belief_states.append(attention_weights)
            timestamps.extend(batch["timestamp"])

            # Also store the target actions for correlation analysis
            actions.append(batch["target"].cpu().numpy())

    # Combine results for analysis
    combined_belief_states = {
        "ohlcv_weights": torch.cat([bs["ohlcv_weights"].cpu() for bs in belief_states]),
        "self_actions_weights": torch.cat(
            [bs["self_actions_weights"].cpu() for bs in belief_states]
        ),
        "other_actions_weights": torch.cat(
            [bs["other_actions_weights"].cpu() for bs in belief_states]
        ),
        "cross_attention": torch.cat(
            [bs["cross_attention"].cpu() for bs in belief_states]
        ),
    }

    combined_actions = np.concatenate(actions)

    return {
        "belief_states": combined_belief_states,
        "timestamps": timestamps,
        "actions": combined_actions,
    }


def visualize_beliefs(belief_analysis, ohlcv_cols, self_action_cols):
    """
    Visualize the belief states and their correlation with actions.

    Args:
        belief_analysis: Dictionary from analyze_beliefs function
        ohlcv_cols: List of OHLCV column names
        self_action_cols: List of agent action column names
    """
    belief_states = belief_analysis["belief_states"]
    timestamps = belief_analysis["timestamps"]
    actions = belief_analysis["actions"]

    # Convert timestamps to datetime if they're not already
    if not isinstance(timestamps[0], datetime):
        timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # 1. Plot variable importance over time
    plt.figure(figsize=(15, 10))

    # OHLCV variable importance
    plt.subplot(3, 1, 1)
    ohlcv_weights = belief_states["ohlcv_weights"].mean(dim=0).squeeze().numpy()
    plt.bar(ohlcv_cols, ohlcv_weights)
    plt.title("OHLCV Feature Importance")
    plt.xticks(rotation=45)

    # Agent actions variable importance
    plt.subplot(3, 1, 2)
    self_weights = belief_states["self_actions_weights"].mean(dim=0).squeeze().numpy()
    plt.bar(self_action_cols, self_weights)
    plt.title("Agent Actions Feature Importance")
    plt.xticks(rotation=90)

    # Cross-attention importance (nested beliefs)
    plt.subplot(3, 1, 3)
    cross_attn = belief_states["cross_attention"].mean(dim=0).numpy()
    plt.imshow(cross_attn, cmap="viridis")
    plt.colorbar()
    plt.title("Cross-Attention Weights (Nested Beliefs)")

    plt.tight_layout()
    plt.savefig("belief_importance.png")
    plt.close()

    # 2. Plot belief evolution over time
    plt.figure(figsize=(15, 8))

    # Track a few important features over time
    top_ohlcv_idx = np.argsort(ohlcv_weights)[-3:]  # Top 3 OHLCV features
    top_agent_idx = np.argsort(self_action_cols)[-3:]  # Top 3 agent action features

    # Plot OHLCV attention over time
    plt.subplot(2, 1, 1)
    for idx in top_ohlcv_idx:
        plt.plot(
            timestamps,
            belief_states["ohlcv_weights"][:, idx].numpy(),
            label=ohlcv_cols[idx],
        )
    plt.title("Top OHLCV Feature Attention Over Time")
    plt.legend()
    plt.xticks(rotation=45)

    # Plot agent action attention over time
    plt.subplot(2, 1, 2)
    for idx in top_agent_idx:
        plt.plot(
            timestamps,
            belief_states["agent_actions_weights"][:, idx].numpy(),
            label=self_action_cols[idx],
        )
    plt.title("Top Agent Action Attention Over Time")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("belief_evolution.png")
    plt.close()

    # 3. Correlation between beliefs and actions
    plt.figure(figsize=(12, 10))

    # Calculate correlation between variable importance and action magnitude
    action_magnitude = np.sum(actions, axis=1)

    # OHLCV correlation
    ohlcv_corr = np.zeros(len(ohlcv_cols))
    for i in range(len(ohlcv_cols)):
        ohlcv_corr[i] = np.corrcoef(
            belief_states["ohlcv_weights"][:, i].numpy(), action_magnitude
        )[0, 1]

    plt.subplot(2, 1, 1)
    plt.bar(ohlcv_cols, ohlcv_corr)
    plt.title("Correlation between OHLCV Attention and Action Magnitude")
    plt.xticks(rotation=45)

    # Agent action correlation
    agent_corr = np.zeros(len(self_action_cols))
    for i in range(len(self_action_cols)):
        agent_corr[i] = np.corrcoef(
            belief_states["agent_actions_weights"][:, i].numpy(), action_magnitude
        )[0, 1]

    plt.subplot(2, 1, 2)
    plt.bar(self_action_cols, agent_corr)
    plt.title("Correlation between Agent Action Attention and Action Magnitude")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig("belief_action_correlation.png")
    plt.close()
