"""
Neural network architectures for stock trading.

This module provides custom neural networks for reinforcement learning,
including multi-scale LSTM architectures for time series processing.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.logging_config import get_logger

logger = get_logger(__name__)


class MultiScaleActorCritic(nn.Module):
    """
    A custom network using multiple LSTM branches with different sequence lengths
    to capture multi-scale temporal patterns.
    
    Architecture:
    - Multiple LSTM branches (one per sequence length)
    - Actor head: FC layers with softmax output
    - Critic head: FC layers with single value output
    
    Args:
        input_dim: Number of input features per timestep
        actor_hidden_dim: Hidden dimension for actor network
        critic_hidden_dim: Hidden dimension for critic network
        action_dim: Number of possible actions
        sequence_lengths: List of sequence lengths for LSTM branches
        num_lstm_layers: Number of LSTM layers per branch
    """

    def __init__(
        self,
        input_dim: int,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
        action_dim: int,
        sequence_lengths: List[int],
        num_lstm_layers: int = 2,
        continuous_actions: bool = False,
        num_stocks: int = 1,
        embedding_dim: int = 16,
    ):
        super().__init__()
        
        self.sequence_lengths = sequence_lengths
        self.continuous_actions = continuous_actions
        self.num_stocks = num_stocks
        self.embedding_dim = embedding_dim
        logger.info(f"Creating MultiScaleActorCritic with sequence lengths: {sequence_lengths}, num_stocks: {num_stocks}, embedding_dim: {embedding_dim}")

        # Stock ID embedding layer
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)

        # Create one LSTM branch per sequence length
        self.lstm_branches = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim,
                hidden_size=actor_hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=0.1 if num_lstm_layers > 1 else 0,
            )
            for _ in sequence_lengths
        ])

        # Total dimension after concatenating LSTM outputs
        total_lstm_out_dim = len(sequence_lengths) * actor_hidden_dim
        
        # Combined dimension after concatenating LSTM features with stock embedding
        combined_dim = total_lstm_out_dim + embedding_dim

        # Normalization layer to stabilize concatenated features
        self.feature_norm = nn.LayerNorm(total_lstm_out_dim)

        # Actor network (now expects combined_dim)
        self.act_fc1 = nn.Linear(combined_dim, actor_hidden_dim)
        if self.continuous_actions:
            self.actor_out_dim = 3
            self.act_fc2 = nn.Linear(actor_hidden_dim, self.actor_out_dim)
            # Separate state-independent learnable parameter for log_std
            # We'll use a parameter that will be passed through softplus to ensure positivity
            self.log_std_logits = nn.Parameter(torch.zeros(self.actor_out_dim))
        else:
            self.actor_out_dim = action_dim
            self.act_fc2 = nn.Linear(actor_hidden_dim, self.actor_out_dim)
        
        # Critic network (now expects combined_dim)
        self.crit_fc1 = nn.Linear(combined_dim, critic_hidden_dim // 2)
        self.crit_fc2 = nn.Linear(critic_hidden_dim // 2, 1)

        # Expected dimensions for Stable-Baselines3 compatibility
        self.latent_dim_pi = self.actor_out_dim
        self.latent_dim_vf = 1
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _extract_features(self, state: torch.Tensor, stock_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features from state using all LSTM branches and combine with stock embedding.
        
        Args:
            state: Input tensor of shape (batch_size, max_seq_len, input_dim)
            stock_id: Optional tensor of shape (batch_size,) with stock IDs. If None, uses 0.
        
        Returns:
            Concatenated features from all branches combined with stock embedding
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        lstm_outputs = []
        for branch, seq_length in zip(self.lstm_branches, self.sequence_lengths):
            # Use the last seq_length timesteps
            state_slice = state[:, -seq_length:, :]
            lstm_out, _ = branch(state_slice)
            # Take final timestep output
            lstm_outputs.append(lstm_out[:, -1, :])
        
        # Concatenate LSTM features
        lstm_features = torch.cat(lstm_outputs, dim=1)
        lstm_features = self.feature_norm(lstm_features)
        
        # Get stock embedding
        if stock_id is None:
            # Default to stock_id 0 if not provided
            batch_size = lstm_features.shape[0]
            stock_id = torch.zeros(batch_size, dtype=torch.long, device=lstm_features.device)
        elif stock_id.dim() == 0:
            # Handle scalar stock_id
            stock_id = stock_id.unsqueeze(0)
        
        # Ensure stock_id is on the same device as features
        stock_id = stock_id.to(lstm_features.device)
        # Clamp stock_id to valid range [0, num_stocks-1] to prevent embedding index errors
        if self.num_stocks > 0:
            stock_id = torch.clamp(stock_id, 0, self.num_stocks - 1)
        embedded_id = self.stock_embedding(stock_id)
        
        # Concatenate LSTM features with stock embedding
        combined_features = torch.cat([lstm_features, embedded_id], dim=-1)
        return combined_features

    def forward(self, features: torch.Tensor, stock_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy and value outputs.
        
        Args:
            features: Input features (state tensor)
            stock_id: Optional stock ID tensor
        
        Returns:
            Tuple of (latent_policy, latent_value)
        """
        return self.forward_actor(features, stock_id), self.forward_critic(features, stock_id)

    def forward_actor(self, state: torch.Tensor, stock_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through actor network.
        
        Args:
            state: Input state tensor
            stock_id: Optional stock ID tensor
        
        Returns:
            Action mean (continuous) or probabilities (discrete)
        """
        x = self._extract_features(state, stock_id)
        x = F.relu(self.act_fc1(x))
        if self.continuous_actions:
            # Output shape (batch, 3) with values strictly between -1 and 1
            x = torch.tanh(self.act_fc2(x))
        else:
            # Add small epsilon to prevent log(0) which causes NaN gradients
            x = F.softmax(self.act_fc2(x), dim=1) + 1e-8
        return x

    def get_action_std(self) -> torch.Tensor:
        """
        Get the standard deviation for continuous actions.
        Ensures positivity using Softplus as requested.
        
        Returns:
            Positive standard deviation tensor of shape (3,)
        """
        if not self.continuous_actions:
            raise ValueError("Standard deviation only available for continuous actions")
        return F.softplus(self.log_std_logits)

    def forward_critic(self, state: torch.Tensor, stock_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state: Input state tensor
            stock_id: Optional stock ID tensor
        
        Returns:
            Value estimate
        """
        x = self._extract_features(state, stock_id)
        x = F.relu(self.crit_fc1(x))
        x = self.crit_fc2(x)
        return x


class SimpleMLP(nn.Module):
    """
    Simple MLP network for baseline comparisons.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.latent_dim_pi = output_dim
        self.latent_dim_vf = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)

