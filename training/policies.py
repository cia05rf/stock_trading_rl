"""
Custom policies for Stable-Baselines3.

This module provides custom ActorCritic policies that integrate with
Stable-Baselines3's PPO implementation while using custom network architectures.
"""

from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from training.networks import MultiScaleActorCritic
from shared.logging_config import get_logger

logger = get_logger(__name__)


class MultiScaleActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy using a MultiScaleActorCritic network.
    
    This policy preserves the 2D observation structure (window_size x features)
    instead of flattening it, allowing the LSTM branches to process temporal
    patterns at multiple scales.
    
    Args:
        observation_space: Observation space (expected to be Box with 2D shape)
        action_space: Action space
        lr_schedule: Learning rate schedule function
        input_dim: Number of input features
        actor_hidden_dim: Hidden dimension for actor
        critic_hidden_dim: Hidden dimension for critic
        action_dim: Number of actions
        sequence_lengths: List of sequence lengths for LSTM branches
        num_lstm_layers: Number of LSTM layers per branch
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        input_dim: int,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
        action_dim: int,
        sequence_lengths: List[int],
        num_lstm_layers: int = 2,
        num_stocks: int = 1,
        embedding_dim: int = 16,
        *args,
        **kwargs,
    ):
        # Check if image space (not supported)
        if is_image_space(observation_space):
            logger.warning("Image processing is not supported by this custom network")
        
        # Extract custom kwargs before passing to parent
        obs_window_size = kwargs.pop('obs_window_size', None)
        n_features = kwargs.pop('n_features', None)
        
        # Store original observation shape
        self.obs_shape = observation_space.shape
        
        # Handle flattened observation space: (obs_window_size * n_features,)
        # We need to infer obs_window_size and n_features
        if len(self.obs_shape) == 1:
            # Flattened observation space - need to reshape to (obs_window_size, n_features)
            flattened_size = self.obs_shape[0]
            
            # Use provided values or infer from flattened size
            if obs_window_size is not None and n_features is not None:
                # Verify the size matches
                if flattened_size != obs_window_size * n_features:
                    logger.warning(
                        f"Provided obs_window_size={obs_window_size} and n_features={n_features} "
                        f"don't match flattened size {flattened_size}. Inferring from size."
                    )
                    obs_window_size = None  # Force inference
                    n_features = None
            
            if obs_window_size is None or n_features is None:
                # Try to infer from flattened size
                if flattened_size % 9 == 0:
                    n_features = 9
                    obs_window_size = flattened_size // 9
                elif flattened_size % 8 == 0:
                    n_features = 8
                    obs_window_size = flattened_size // 8
                else:
                    # Default fallback
                    logger.warning(
                        f"Cannot infer obs_window_size and n_features from "
                        f"flattened size {flattened_size}. Using defaults."
                    )
                    n_features = 9
                    obs_window_size = 50
            
            self.obs_window_size = obs_window_size
            self.n_features = n_features
            input_dim = n_features  # Features per timestep
            logger.info(
                f"Flattened observation space detected: {self.obs_shape} -> "
                f"({obs_window_size}, {n_features})"
            )
        else:
            # 2D observation space (backward compatibility)
            input_dim = self.obs_shape[-1]  # Number of features
            self.obs_window_size = self.obs_shape[0]
            self.n_features = input_dim

        # Store hyperparameters
        self.input_dim = input_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.action_dim = action_dim
        self.sequence_lengths = sequence_lengths
        self.num_lstm_layers = num_lstm_layers
        self.num_stocks = num_stocks
        self.embedding_dim = embedding_dim
        self.lr_schedule = lr_schedule
        # Check if we are in continuous mode
        self.is_continuous = isinstance(action_space, gym.spaces.Box)
        
        # Store current stock_id (will be updated from environment)
        self.current_stock_id: Optional[torch.Tensor] = None

        # Disable orthogonal initialization (we handle it in the network)
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )
        
        logger.info(f"Initialized MultiScaleActorCriticPolicy with obs_shape={self.obs_shape}, num_stocks={num_stocks}")

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Override to reshape flattened observations to sequence format.
        
        The observation space is now flattened (obs_window_size * n_features,),
        but we need to reshape it to (obs_window_size, n_features) for the LSTM.
        
        Args:
            obs: Flattened observation tensor of shape (batch_size, obs_window_size * n_features)
        
        Returns:
            Reshaped tensor of shape (batch_size, obs_window_size, n_features)
        """
        batch_size = obs.shape[0]
        
        # Reshape from (batch_size, obs_window_size * n_features) to (batch_size, obs_window_size, n_features)
        if hasattr(self, 'obs_window_size') and hasattr(self, 'n_features'):
            return obs.view(batch_size, self.obs_window_size, self.n_features)
        else:
            # Fallback: try to infer from observation space
            if len(self.observation_space.shape) == 1:
                # Flattened: infer dimensions
                flattened_size = self.observation_space.shape[0]
                if flattened_size % 9 == 0:
                    n_features = 9
                    obs_window_size = flattened_size // 9
                else:
                    # Default
                    obs_window_size = 50
                    n_features = flattened_size // obs_window_size
                return obs.view(batch_size, obs_window_size, n_features)
            else:
                # 2D observation space (backward compatibility)
                return obs.view(batch_size, *self.observation_space.shape)

    def _build_mlp_extractor(self) -> None:
        """
        Build the custom MLP extractor (our MultiScaleActorCritic network).
        """
        self.mlp_extractor = MultiScaleActorCritic(
            input_dim=self.input_dim,
            actor_hidden_dim=self.actor_hidden_dim,
            critic_hidden_dim=self.critic_hidden_dim,
            action_dim=self.action_dim,
            sequence_lengths=self.sequence_lengths,
            num_lstm_layers=self.num_lstm_layers,
            continuous_actions=self.is_continuous,
            num_stocks=self.num_stocks,
            embedding_dim=self.embedding_dim,
        )
    
    def set_stock_id(self, stock_id: Optional[int]) -> None:
        """
        Set the current stock_id for the policy.
        
        Args:
            stock_id: Stock ID integer or None
        """
        if stock_id is not None:
            self.current_stock_id = torch.tensor([stock_id], dtype=torch.long)
        else:
            self.current_stock_id = None
    
    def _get_stock_id_tensor(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get stock_id tensor for the current batch.
        
        Args:
            batch_size: Batch size
            device: Device to place tensor on
        
        Returns:
            Stock ID tensor or None
        """
        if self.current_stock_id is not None:
            # Expand to batch size if needed
            stock_id = self.current_stock_id.to(device)
            if stock_id.shape[0] == 1 and batch_size > 1:
                stock_id = stock_id.repeat(batch_size)
            # Clamp stock_id to valid range [0, num_stocks-1] to prevent embedding errors
            if self.num_stocks > 0:
                stock_id = torch.clamp(stock_id, 0, self.num_stocks - 1)
            return stock_id
        return None

    def get_distribution(self, obs: torch.Tensor):
        """
        Override to pass stock_id to mlp_extractor and clamp Gaussian log_std.
        """
        if self.is_continuous and hasattr(self, "log_std") and self.log_std is not None:
            with torch.no_grad():
                # Typical safe range used in many continuous-control implementations.
                self.log_std.data = torch.nan_to_num(
                    self.log_std.data,
                    nan=-2.0,
                    posinf=2.0,
                    neginf=-20.0,
                )
                self.log_std.data.clamp_(-20.0, 2.0)
        
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self._call_mlp_extractor_with_stock_id(features, use_actor=True)
        return self._get_action_dist_from_latent(latent_pi)
    
    def _call_mlp_extractor_with_stock_id(self, features: torch.Tensor, use_actor: bool = False, use_critic: bool = False):
        """
        Call mlp_extractor with stock_id if available.
        
        Args:
            features: Extracted features
            use_actor: Whether to call forward_actor
            use_critic: Whether to call forward_critic
        
        Returns:
            Latent features from actor and/or critic
        """
        batch_size = features.shape[0] if features.dim() > 0 else 1
        device = features.device
        stock_id = self._get_stock_id_tensor(batch_size, device)
        
        if use_actor and use_critic:
            # Call both
            latent_pi = self.mlp_extractor.forward_actor(features, stock_id)
            latent_vf = self.mlp_extractor.forward_critic(features, stock_id)
            return latent_pi, latent_vf
        elif use_actor:
            return self.mlp_extractor.forward_actor(features, stock_id)
        elif use_critic:
            return self.mlp_extractor.forward_critic(features, stock_id)
        else:
            # Call forward (both)
            return self.mlp_extractor(features, stock_id)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Override to pass stock_id to mlp_extractor.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self._call_mlp_extractor_with_stock_id(features, use_actor=True, use_critic=True)
        else:
            pi_features, vf_features = features
            latent_pi = self._call_mlp_extractor_with_stock_id(pi_features, use_actor=True)
            latent_vf = self._call_mlp_extractor_with_stock_id(vf_features, use_critic=True)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Override to pass stock_id to mlp_extractor.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self._call_mlp_extractor_with_stock_id(features, use_actor=True, use_critic=True)
        else:
            pi_features, vf_features = features
            latent_pi = self._call_mlp_extractor_with_stock_id(pi_features, use_actor=True)
            latent_vf = self._call_mlp_extractor_with_stock_id(vf_features, use_critic=True)
        
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
    
    def predict_values(self, obs: torch.Tensor):
        """
        Override to pass stock_id to mlp_extractor.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self._call_mlp_extractor_with_stock_id(features, use_critic=True)
        return self.value_net(latent_vf)


class IdentityFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features extractor that reshapes flattened observations into sequence format.
    
    The new observation space is flattened (obs_window_size * 9,), but the network
    expects a 2D sequence (obs_window_size, 9) for LSTM processing.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        """
        Initialize the identity extractor.
        
        Args:
            observation_space: Original observation space (flattened)
        """
        # Set features_dim to flattened size for compatibility
        features_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=features_dim)
        
        self._observation_shape = observation_space.shape
        
        # Infer obs_window_size and n_features from observation space
        # Default: 9 features per timestep (as per refactor spec)
        n_features = 9
        if features_dim % n_features == 0:
            self.obs_window_size = features_dim // n_features
            self.n_features = n_features
        else:
            # Try other common feature counts
            for nf in [8, 10, 7]:
                if features_dim % nf == 0:
                    self.n_features = nf
                    self.obs_window_size = features_dim // nf
                    break
            else:
                # Fallback: assume 50 timesteps
                self.obs_window_size = 50
                self.n_features = features_dim // 50
                logger.warning(
                    f"Could not cleanly infer obs_window_size and n_features from "
                    f"features_dim={features_dim}. Using obs_window_size={self.obs_window_size}, "
                    f"n_features={self.n_features}"
                )
        
        logger.info(
            f"IdentityFeaturesExtractor: reshaping ({features_dim},) -> "
            f"({self.obs_window_size}, {self.n_features})"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reshape flattened observations to sequence format.
        
        Args:
            observations: Flattened observation tensor of shape (batch_size, obs_window_size * n_features)
        
        Returns:
            Reshaped tensor of shape (batch_size, obs_window_size, n_features)
        """
        batch_size = observations.shape[0]
        # Reshape from (batch_size, obs_window_size * n_features) to (batch_size, obs_window_size, n_features)
        return observations.view(batch_size, self.obs_window_size, self.n_features)

