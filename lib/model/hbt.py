import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefEmbeddingModule(nn.Module):
    """
    Module for embedding beliefs based on different input types.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Projection layer
        self.projection = nn.Linear(input_dim, hidden_dim)

        # Self-attention for capturing relationships within the sequence
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Normalization and feed-forward layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through the belief embedding module.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Project input to hidden dimension
        proj_x = self.projection(x)

        # Self-attention
        attn_output, attn_weights = self.self_attention(
            query=proj_x, key=proj_x, value=proj_x
        )

        # First residual connection and normalization
        x1 = self.norm1(proj_x + attn_output)

        # Feed-forward network
        ff_output = self.feed_forward(x1)

        # Second residual connection and normalization
        output = self.norm2(x1 + ff_output)

        return output, attn_weights


class CrossAttentionModule(nn.Module):
    """
    Cross-attention module for integrating multiple belief embeddings.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value):
        """
        Forward pass through the cross-attention module.

        Args:
            query: Query tensor of shape [batch_size, seq_len, hidden_dim]
            key_value: Key/value tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (output tensor, attention weights)
        """
        output, attn_weights = self.cross_attention(
            query=query, key=key_value, value=key_value
        )

        # Residual connection and normalization
        output = self.norm(query + output)

        return output, attn_weights


class HierarchicalTransformerModel(nn.Module):
    """
    Hierarchical Transformer model with belief embedding modules for market behavior prediction.
    This model implements both the benchmark and proposed model from the experiment setup.
    """

    def __init__(
        self,
        ohlcv_dim=9,
        action_dim=40,
        hidden_dim=64,
        output_dim=40,
        context_length=5,
        num_heads=4,
        dropout=0.1,
        use_other_player_beliefs=True,  # Set to False for benchmark model
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.use_other_player_beliefs = use_other_player_beliefs

        # Positional encoding for sequential data
        self.pos_encoding = nn.Parameter(torch.zeros(1, context_length, hidden_dim))
        nn.init.xavier_uniform_(self.pos_encoding)

        # === Self-belief module (f_i) ===
        # OHLCV belief embedding module
        self.ohlcv_belief_module = BeliefEmbeddingModule(
            input_dim=ohlcv_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Agent's own actions belief embedding module
        self.agent_action_belief_module = BeliefEmbeddingModule(
            input_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # === Other-belief module (g_i) ===
        # Other players' actions belief embedding module
        if use_other_player_beliefs:
            self.other_action_belief_module = BeliefEmbeddingModule(
                input_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

            # Cross-attention for integrating beliefs b_i^t(k)
            self.cross_attention = CrossAttentionModule(
                hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
            )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Softmax(dim=-1),  # For distribution output
        )

    def forward(self, ohlcv, agent_actions, other_actions=None):
        """
        Forward pass through the hierarchical transformer model.

        Args:
            ohlcv: OHLCV data tensor of shape [batch_size, seq_len, ohlcv_dim]
            agent_actions: Agent's own action distribution tensor of shape [batch_size, seq_len, action_dim]
            other_actions: Other players' action distribution tensor of shape [batch_size, seq_len, action_dim]

        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size = ohlcv.shape[0]

        # Add positional encoding to inputs (broadcast across batch dimension)
        pos_enc = self.pos_encoding.repeat(batch_size, 1, 1)

        # === Self-belief embedding (f_i) ===
        # Process OHLCV data
        ohlcv_belief, ohlcv_weights = self.ohlcv_belief_module(ohlcv)
        ohlcv_belief = ohlcv_belief + pos_enc

        # Process agent's own actions
        agent_belief, agent_weights = self.agent_action_belief_module(agent_actions)
        agent_belief = agent_belief + pos_enc

        # Combine self-beliefs from OHLCV and agent's own actions
        # Taking the last step in the sequence (most recent)
        self_belief = ohlcv_belief[:, -1, :] + agent_belief[:, -1, :]
        self_belief = self_belief.unsqueeze(
            1
        )  # Add sequence dimension back [batch, 1, hidden]

        # For the benchmark model, we use only self-belief
        combined_belief = self_belief
        cross_attn_weights = None
        other_weights = None

        # === Other-belief embedding (g_i) ===
        if self.use_other_player_beliefs and other_actions is not None:
            # Process other players' actions
            other_belief, other_weights = self.other_action_belief_module(other_actions)
            other_belief = other_belief + pos_enc

            # Take the last step for other players' belief
            other_belief = other_belief[:, -1, :].unsqueeze(1)

            # Cross-attention to integrate self and other beliefs
            combined_belief, cross_attn_weights = self.cross_attention(
                self_belief, other_belief
            )

        # Predict next action distribution
        # Use the combined belief to predict the next action
        predictions = self.output_layer(combined_belief.squeeze(1))

        # Collect attention weights for belief analysis
        attention_weights = {
            "ohlcv_weights": ohlcv_weights,
            "agent_actions_weights": agent_weights,
            "other_actions_weights": other_weights,
            "cross_attention": cross_attn_weights,
        }

        # softmax 적용
        predictions = F.softmax(predictions, dim=-1)

        return predictions, attention_weights


class HierarchicalModelTrainer:
    """
    Trainer class for the Hierarchical Transformer model.
    """

    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, ohlcv, agent_actions, other_actions, targets):
        """
        Perform a single training step.

        Args:
            ohlcv: OHLCV data tensor
            agent_actions: Agent's own action distribution tensor
            other_actions: Other players' action distribution tensor
            targets: Target action distribution tensor

        Returns:
            Loss value
        """
        self.optimizer.zero_grad()

        predictions, _ = self.model(ohlcv, agent_actions, other_actions)

        # Calculate KL divergence loss
        loss = F.kl_div(torch.log(predictions + 1e-10), torch.log(targets + 1e-10), reduction="batchmean")

        # NaN 값 검사 및 로깅
        if torch.isnan(loss):
            print(f"[경고] NaN 손실 값 발견: {loss.item()}")
        
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, ohlcv, agent_actions, other_actions, targets):
        """
        Evaluate the model on validation or test data.

        Args:
            ohlcv: OHLCV data tensor
            agent_actions: Agent's own action distribution tensor
            other_actions: Other players' action distribution tensor
            targets: Target action distribution tensor

        Returns:
            Tuple of (loss value, predictions)
        """
        predictions, attention_weights = self.model(ohlcv, agent_actions, other_actions)

        # Calculate KL divergence loss
        loss = F.kl_div(torch.log(predictions + 1e-10), torch.log(targets + 1e-10), reduction="batchmean")

        # NaN 값 검사 및 로깅
        if torch.isnan(loss):
            print(f"[경고] NaN 손실 값 발견: {loss.item()}")
            
        return loss.item(), attention_weights

    def save_model(self, filepath):
        """Save model to file."""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load model from file."""
        self.model.load_state_dict(torch.load(filepath))


def create_benchmark_model(ohlcv_dim=9, action_dim=40, hidden_dim=64, context_length=5):
    """
    Create the benchmark model without other-player belief embeddings.

    Args:
        ohlcv_dim: Dimension of OHLCV data
        action_dim: Dimension of action distribution
        hidden_dim: Hidden dimension size
        context_length: Number of time steps in context window

    Returns:
        Benchmark model instance
    """
    return HierarchicalTransformerModel(
        ohlcv_dim=ohlcv_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        output_dim=action_dim,
        context_length=context_length,
        use_other_player_beliefs=False,  # Benchmark doesn't use other player beliefs
    )


def create_proposed_model(ohlcv_dim=9, action_dim=40, hidden_dim=64, context_length=5):
    """
    Create the proposed model with other-player belief embeddings.

    Args:
        ohlcv_dim: Dimension of OHLCV data
        action_dim: Dimension of action distribution
        hidden_dim: Hidden dimension size
        context_length: Number of time steps in context window

    Returns:
        Proposed model instance
    """
    return HierarchicalTransformerModel(
        ohlcv_dim=ohlcv_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        output_dim=action_dim,
        context_length=context_length,
        use_other_player_beliefs=True,  # Proposed model uses other player beliefs
    )
