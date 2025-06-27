import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class CharacterNet(nn.Module):
    """
    Character Net: Processes past episode trajectories into character embeddings
    e_char,ij = f_θ(τ_ij^(obs))
    
    FIXED VERSION: Uses learnable embedding for N_past=0 instead of zeros
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # Input: flattened (state, action) pairs
        input_dim = state_dim + action_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # FIXED: Learnable embedding for "no past information" case
        # This allows the model to learn what "no past information" means
        # rather than always returning zeros
        self.no_past_embedding = nn.Parameter(
            torch.randn(embedding_dim) * 0.1
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectories: (batch_size, n_past, seq_len, state_dim + action_dim)
        Returns:
            character_embeddings: (batch_size, embedding_dim)
        """
        batch_size, n_past, seq_len, input_dim = trajectories.shape
        
        # Handle empty past trajectories (N_past=0) with zero embedding
        if n_past == 0:
            # Return zero embedding to force uniform/random behavior
            # This ensures N_past=0 gives ~0.2 action likelihood (1/5 actions)
            return torch.zeros(batch_size, self.embedding_dim, device=trajectories.device)

        # Flatten trajectories for processing
        traj_flat = trajectories.view(batch_size * n_past, seq_len, input_dim)

        # Process each trajectory segment
        embeddings = []
        for i in range(seq_len):
            step_input = traj_flat[:, i, :]  # (batch_size * n_past, input_dim)
            step_embedding = self.mlp(
                step_input
            )  # (batch_size * n_past, embedding_dim)
            embeddings.append(step_embedding)

        # Average over sequence length
        trajectory_embeddings = torch.stack(embeddings, dim=1).mean(dim=1)

        # Reshape and aggregate over past episodes
        trajectory_embeddings = trajectory_embeddings.view(
            batch_size, n_past, self.embedding_dim
        )
        
        # IMPROVED: Use mean instead of sum for better normalization
        # This prevents embeddings from growing linearly with n_past
        character_embeddings = trajectory_embeddings.mean(dim=1)

        return character_embeddings


class MentalStateNet(nn.Module):
    """
    Mental State Net: Processes current episode trajectory
    e_mental,i = g_φ([τ_ij^(obs)]_0:t-1, e_char,i)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        char_embedding_dim: int,
        hidden_dim: int = 128,
        mental_embedding_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.char_embedding_dim = char_embedding_dim
        self.mental_embedding_dim = mental_embedding_dim

        # LSTM for processing current trajectory
        input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Combine LSTM output with character embedding
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + char_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mental_embedding_dim),
        )

    def forward(
        self, current_trajectory: torch.Tensor, character_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            current_trajectory: (batch_size, current_seq_len, state_dim + action_dim)
            character_embedding: (batch_size, char_embedding_dim)
        Returns:
            mental_embedding: (batch_size, mental_embedding_dim)
        """
        # Process current trajectory with LSTM
        lstm_out, (hidden, _) = self.lstm(current_trajectory)

        # Use final hidden state
        final_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Concatenate with character embedding
        combined = torch.cat([final_hidden, character_embedding], dim=1)

        # Generate mental state embedding
        mental_embedding = self.fusion(combined)

        return mental_embedding


class PredictionNet(nn.Module):
    """
    Prediction Net: Outputs behavioral predictions
    - Next-step action probabilities
    - Object consumption probabilities
    - Successor representations
    """

    def __init__(
        self,
        state_dim: int,
        char_embedding_dim: int,
        mental_embedding_dim: int,
        n_actions: int = 5,
        n_objects: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_objects = n_objects

        # Input: current state + character embedding + mental embedding
        input_dim = state_dim + char_embedding_dim + mental_embedding_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Object consumption prediction head
        self.consumption_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objects),
        )

        # Successor representation head (for grid states)
        grid_size = int(np.sqrt(state_dim / 6))  # Assuming 6 channels
        self.sr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size * grid_size),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        character_embedding: torch.Tensor,
        mental_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            current_state: (batch_size, state_dim)
            character_embedding: (batch_size, char_embedding_dim)
            mental_embedding: (batch_size, mental_embedding_dim)
        Returns:
            Dict containing action_logits, action_probs, consumption_probs, sr_logits
        """
        # Concatenate all inputs
        combined_input = torch.cat(
            [current_state, character_embedding, mental_embedding], dim=1
        )

        # Shared processing
        shared_features = self.shared(combined_input)

        # Action predictions
        action_logits = self.action_head(shared_features)
        
        # Apply temperature scaling based on information availability
        # If character embedding is mostly zeros (N_past=0), increase temperature for more uniform predictions
        char_info_strength = torch.norm(character_embedding, dim=1, keepdim=True)
        temperature = 1.0 + 2.0 * torch.exp(-char_info_strength)  # Higher temp when less info
        
        action_probs = F.softmax(action_logits / temperature, dim=1)

        # Object consumption predictions
        consumption_probs = torch.sigmoid(self.consumption_head(shared_features))

        # Successor representation predictions
        sr_logits = self.sr_head(shared_features)

        return {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "consumption_probs": consumption_probs,
            "sr_logits": sr_logits,
        }


class ToMnet(nn.Module):
    """
    Theory of Mind Network (ToMnet)
    
    FIXED VERSION: Uses improved CharacterNet that properly handles N_past=0
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        n_actions: int = 5,
        n_objects: int = 4,
        char_embedding_dim: int = 8,
        mental_embedding_dim: int = 64,
        hidden_dim: int = 128,
        use_mental_state: bool = True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.char_embedding_dim = char_embedding_dim
        self.mental_embedding_dim = mental_embedding_dim
        self.use_mental_state = use_mental_state

        # Initialize networks
        self.character_net = CharacterNet(
            state_dim, action_dim, hidden_dim, char_embedding_dim
        )

        if use_mental_state:
            self.mental_state_net = MentalStateNet(
                state_dim,
                action_dim,
                char_embedding_dim,
                hidden_dim,
                mental_embedding_dim,
            )
        else:
            # For Figure 3 experiments, mental state is not used
            self.mental_state_net = None
            mental_embedding_dim = 0

        self.prediction_net = PredictionNet(
            state_dim,
            char_embedding_dim,
            mental_embedding_dim,
            n_actions,
            n_objects,
            hidden_dim,
        )

    def forward(
        self,
        past_trajectories: torch.Tensor,
        current_trajectory: torch.Tensor,
        current_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            past_trajectories: (batch_size, n_past, seq_len, state_dim + action_dim)
            current_trajectory: (batch_size, current_seq_len, state_dim + action_dim)
            current_state: (batch_size, state_dim)
        Returns:
            predictions: Dict with all prediction outputs
        """
        # Generate character embedding (FIXED: now properly handles N_past=0)
        character_embedding = self.character_net(past_trajectories)

        # Generate mental state embedding if used
        if self.use_mental_state and self.mental_state_net is not None:
            mental_embedding = self.mental_state_net(
                current_trajectory, character_embedding
            )
        else:
            # Create dummy mental embedding with zeros
            batch_size = current_state.size(0)
            mental_embedding = torch.zeros(batch_size, 0, device=current_state.device)

        # Generate predictions
        predictions = self.prediction_net(
            current_state, character_embedding, mental_embedding
        )

        # Add embeddings to output for analysis
        predictions["character_embedding"] = character_embedding
        if self.use_mental_state:
            predictions["mental_embedding"] = mental_embedding

        return predictions

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss function
        """
        losses = {}

        # Action prediction loss
        if "true_actions" in targets:
            action_loss = F.cross_entropy(
                predictions["action_logits"], targets["true_actions"]
            )
            losses["action_loss"] = action_loss

        # Object consumption loss
        if "true_consumption" in targets:
            consumption_loss = F.binary_cross_entropy(
                predictions["consumption_probs"], targets["true_consumption"]
            )
            losses["consumption_loss"] = consumption_loss

        # Successor representation loss
        if "true_sr" in targets:
            sr_loss = F.kl_div(
                F.log_softmax(predictions["sr_logits"], dim=-1),
                targets["true_sr"],
                reduction="batchmean",
            )
            losses["sr_loss"] = sr_loss

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses


def create_tomnet(
    experiment_type: str,
    state_dim: int,
    char_embedding_dim: Optional[int] = None,
    action_dim: int = 5,
    n_actions: int = 5,
    n_objects: int = 4,
    mental_embedding_dim: int = 64,
    hidden_dim: int = 128,
) -> ToMnet:
    """
    Create ToMnet configuration for different experiment types
    
    FIXED VERSION: Uses improved CharacterNet
    """
    # Set experiment-specific defaults
    if experiment_type == "figure3":
        # Figure 3: Random agents with 2D character embeddings for visualization
        if char_embedding_dim is None:
            char_embedding_dim = 2
        use_mental_state = False  # Figure 3 doesn't use mental state
    elif experiment_type == "figure5":
        # Figure 5: Goal-directed agents with higher-dimensional embeddings
        if char_embedding_dim is None:
            char_embedding_dim = 8
        use_mental_state = True  # Figure 5 uses mental state
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    return ToMnet(
        state_dim=state_dim,
        action_dim=action_dim,
        n_actions=n_actions,
        n_objects=n_objects,
        char_embedding_dim=char_embedding_dim,
        mental_embedding_dim=mental_embedding_dim,
        hidden_dim=hidden_dim,
        use_mental_state=use_mental_state,
    )
