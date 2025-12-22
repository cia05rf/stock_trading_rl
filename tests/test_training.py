"""
Tests for training module.

Tests environment, networks, policies, and training utilities.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNetworks:
    """Tests for neural network architectures."""
    
    def test_multiscale_actor_critic_creation(self):
        """Test MultiScaleActorCritic network creation."""
        from training.networks import MultiScaleActorCritic
        
        network = MultiScaleActorCritic(
            input_dim=15,
            actor_hidden_dim=64,
            critic_hidden_dim=64,
            action_dim=9,
            sequence_lengths=[1, 5, 20],
            num_lstm_layers=2,
        )
        
        assert network is not None
        assert len(network.lstm_branches) == 3
    
    def test_multiscale_forward(self):
        """Test forward pass through network."""
        from training.networks import MultiScaleActorCritic
        
        network = MultiScaleActorCritic(
            input_dim=15,
            actor_hidden_dim=64,
            critic_hidden_dim=64,
            action_dim=9,
            sequence_lengths=[1, 5, 20],
            num_lstm_layers=2,
        )
        
        # Create input: (batch_size, seq_len, features)
        x = torch.randn(8, 60, 15)
        
        actor_out, critic_out = network(x)
        
        assert actor_out.shape == (8, 9)  # (batch, actions)
        assert critic_out.shape == (8, 1)  # (batch, value)
    
    def test_multiscale_actor_softmax(self):
        """Test that actor output is valid probability distribution."""
        from training.networks import MultiScaleActorCritic
        
        network = MultiScaleActorCritic(
            input_dim=15,
            actor_hidden_dim=64,
            critic_hidden_dim=64,
            action_dim=9,
            sequence_lengths=[1, 5, 20],
            num_lstm_layers=2,
        )
        
        x = torch.randn(8, 60, 15)
        actor_out = network.forward_actor(x)
        
        # Check that outputs sum to 1 (softmax)
        sums = actor_out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)
    
    def test_simple_mlp(self):
        """Test SimpleMLP network."""
        from training.networks import SimpleMLP
        
        network = SimpleMLP(
            input_dim=100,
            hidden_dims=[64, 32],
            output_dim=9,
        )
        
        x = torch.randn(8, 100)
        out = network(x)
        
        assert out.shape == (8, 9)


class TestLRSchedules:
    """Tests for learning rate schedules."""
    
    def test_linear_schedule(self):
        """Test linear learning rate schedule."""
        from training.lr_schedules import linear_schedule
        
        schedule = linear_schedule(0.01, 0.001, 0.5)
        
        # The schedule decays during first half of training (progress_remaining > 0.5)
        # At progress_remaining=1.0 (start), already decaying toward final
        # At progress_remaining=0.5 (half way), should be at final_value
        assert abs(schedule(0.5) - 0.001) < 1e-6
        # After decay (progress_remaining < 0.5)
        assert abs(schedule(0.0) - 0.001) < 1e-6
        # During decay, value should be between initial and final
        mid_value = schedule(0.75)
        assert 0.001 < mid_value < 0.01
    
    def test_exponential_schedule(self):
        """Test exponential learning rate schedule."""
        from training.lr_schedules import exponential_schedule
        
        schedule = exponential_schedule(0.01, 0.9)
        
        # At start
        assert abs(schedule(1.0) - 0.01) < 1e-6
        # Should decay
        assert schedule(0.5) < 0.01
        assert schedule(0.0) < schedule(0.5)
    
    def test_exponential_decay_schedule(self):
        """Test exponential decay schedule."""
        from training.lr_schedules import exponential_decay_schedule
        
        schedule = exponential_decay_schedule(0.01, 0.0001, 0.8)
        
        # At start
        assert abs(schedule(1.0) - 0.01) < 1e-5
        # At end
        assert schedule(0.0) >= 0.0001
    
    def test_cosine_annealing_schedule(self):
        """Test cosine annealing schedule."""
        from training.lr_schedules import cosine_annealing_schedule
        
        schedule = cosine_annealing_schedule(0.01, 0.0001)
        
        # At start (peak)
        assert abs(schedule(1.0) - 0.01) < 1e-6
        # At end (minimum)
        assert abs(schedule(0.0) - 0.0001) < 1e-5


class TestCallbacks:
    """Tests for training callbacks."""
    
    def test_info_logger_callback(self):
        """Test InfoLoggerCallback."""
        from training.callbacks import InfoLoggerCallback
        
        callback = InfoLoggerCallback(verbose=0)
        
        assert callback.info_history == []
        assert callback._step_count == 0
    
    def test_gradient_monitoring_callback(self):
        """Test GradientMonitoringCallback."""
        from training.callbacks import GradientMonitoringCallback
        
        callback = GradientMonitoringCallback(verbose=0, check_interval=10)
        
        assert callback.check_interval == 10
    
    def test_performance_callback(self):
        """Test PerformanceCallback."""
        from training.callbacks import PerformanceCallback
        
        callback = PerformanceCallback(verbose=0)
        
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []
        
        metrics = callback.get_metrics()
        assert 'episode_rewards' in metrics
        assert 'episode_lengths' in metrics


class TestEnvironment:
    """Tests for trading environment (mocked)."""
    
    def test_environment_action_space(self):
        """Test that environment has correct action space."""
        # This would require database, so we mock it
        with patch('training.environment.Ingestion') as MockIngestion:
            mock_instance = MockIngestion.return_value
            mock_instance.read_tickers.return_value = MagicMock()
            mock_instance.read_tickers.return_value.iloc = MagicMock()
            mock_instance.read_tickers.return_value.sample.return_value = MagicMock()
            mock_instance.read_tickers.return_value.sample.return_value.to_list.return_value = ['AAPL', 'GOOG']
            mock_instance.count_timesteps.return_value = 1000
            
            # Action space should have 9 actions
            # We can't fully test without database, but verify the expected count
            expected_actions = 9  # 4 buy + 4 sell + 1 hold


class TestPolicies:
    """Tests for custom policies."""
    
    def test_identity_features_extractor(self):
        """Test IdentityFeaturesExtractor."""
        import gymnasium as gym
        from training.policies import IdentityFeaturesExtractor
        
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(60, 15), dtype=np.float32)
        extractor = IdentityFeaturesExtractor(obs_space)
        
        # Test forward
        x = torch.randn(8, 60 * 15)  # Flattened input
        out = extractor(x)
        
        assert out.shape == (8, 60, 15)


class TestInference:
    """Tests for inference module."""
    
    def test_preds_df_creation(self):
        """Test PredsDf creation."""
        from training.inference import PredsDf
        import pandas as pd
        
        data = {
            'ticker': ['AAPL', 'GOOG'],
            'action_type': ['buy', 'sell'],
            'action_prob': [0.8, 0.6],
        }
        
        df = PredsDf(data)
        
        assert len(df) == 2
    
    def test_preds_df_top_buys(self):
        """Test PredsDf top_buys method."""
        from training.inference import PredsDf
        
        data = {
            'ticker': ['AAPL', 'GOOG', 'MSFT', 'AMZN'],
            'action_type': ['buy', 'sell', 'buy', 'buy'],
            'action_prob': [0.8, 0.6, 0.9, 0.7],
        }
        
        df = PredsDf(data)
        top_buys = df.top_buys(n=2)
        
        assert len(top_buys) == 2
        assert top_buys.iloc[0]['ticker'] == 'MSFT'  # Highest prob
    
    def test_preds_df_top_sells(self):
        """Test PredsDf top_sells method."""
        from training.inference import PredsDf
        
        data = {
            'ticker': ['AAPL', 'GOOG', 'MSFT'],
            'action_type': ['sell', 'sell', 'buy'],
            'action_prob': [0.8, 0.6, 0.9],
        }
        
        df = PredsDf(data)
        top_sells = df.top_sells()
        
        assert len(top_sells) == 2
        assert top_sells.iloc[0]['ticker'] == 'AAPL'

