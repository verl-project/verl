"""
Tests for the Weighted SFT Interface

This module contains comprehensive tests for the weighted SFT interface
to ensure correct loss computation, advantage weighting, and masking.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atropos_sft_interface import WeightedSFTInterface, AtroposBatchProcessor
from atroposlib.type_definitions import WeightedSFTBatch


class TestWeightedSFTInterface:
    """Test cases for the WeightedSFTInterface class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            "loss_reduction": "mean",
            "ignore_index": -100,
            "advantage_normalization": "none",
            "temperature": 1.0
        }
        self.interface = WeightedSFTInterface(self.config)
        
        # Create sample data
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        
        # Sample logits, tokens, masks, advantages
        torch.manual_seed(42)
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.loss_masks = torch.ones(self.batch_size, self.seq_len)
        self.advantages = torch.ones(self.batch_size, self.seq_len)
    
    def test_basic_loss_computation(self):
        """Test basic loss computation without advantages."""
        # Set advantages to 1.0 (no weighting)
        advantages = torch.ones_like(self.advantages)
        
        result = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=advantages
        )
        
        assert "loss" in result
        assert "token_losses" in result
        assert "weighted_losses" in result
        assert "effective_mask" in result
        
        # Loss should be a scalar
        assert result["loss"].dim() == 0
        
        # Token losses should match expected shape
        expected_shape = (self.batch_size, self.seq_len - 1)  # -1 due to label shifting
        assert result["token_losses"].shape == expected_shape
    
    def test_advantage_weighting(self):
        """Test that advantages properly weight the loss."""
        # Test with different advantage values
        advantages_high = torch.ones_like(self.advantages) * 2.0
        advantages_low = torch.ones_like(self.advantages) * 0.5
        
        result_high = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=advantages_high
        )
        
        result_low = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=advantages_low
        )
        
        # Higher advantages should lead to higher loss
        assert result_high["loss"].item() > result_low["loss"].item()
    
    def test_loss_masking(self):
        """Test that loss masking works correctly."""
        # Create mask that excludes some tokens
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[0, 1] = 0  # Exclude second token of first sequence
        mask[1, 2] = 0  # Exclude third token of second sequence
        
        result = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=mask,
            advantages=self.advantages
        )
        
        # Check that effective mask reflects the masking
        effective_mask = result["effective_mask"]
        assert effective_mask[0, 0] == 1.0  # Should be included (after shifting)
        assert effective_mask[1, 1] == 1.0  # Should be included (after shifting)
    
    def test_ignore_index(self):
        """Test that ignore_index is properly handled."""
        # Set some tokens to ignore_index
        tokens = self.tokens.clone()
        tokens[0, 1] = self.config["ignore_index"]
        tokens[1, 2] = self.config["ignore_index"]
        
        result = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=tokens,
            loss_masks=self.loss_masks,
            advantages=self.advantages
        )
        
        # Tokens with ignore_index should be masked out
        effective_mask = result["effective_mask"]
        # Note: due to label shifting, ignore_index at position i affects position i-1 in the mask
        assert effective_mask[0, 0] == 0.0  # Should be masked (ignore_index at pos 1 -> affects pos 0)
        assert effective_mask[1, 1] == 0.0  # Should be masked (ignore_index at pos 2 -> affects pos 1)
    
    def test_sequence_level_advantages(self):
        """Test handling of sequence-level advantages."""
        # Provide sequence-level advantages (one per sequence)
        seq_advantages = torch.tensor([1.5, 0.5])
        
        result = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=seq_advantages
        )
        
        # Should broadcast to token level
        assert result["advantages"].shape == (self.batch_size, self.seq_len - 1)
        assert torch.allclose(result["advantages"][0], torch.tensor(1.5))
        assert torch.allclose(result["advantages"][1], torch.tensor(0.5))
    
    def test_advantage_normalization(self):
        """Test advantage normalization options."""
        # Test batch normalization
        config_batch_norm = self.config.copy()
        config_batch_norm["advantage_normalization"] = "batch"
        interface_batch = WeightedSFTInterface(config_batch_norm)
        
        advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        
        result = interface_batch.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=advantages
        )
        
        # Normalized advantages should have approximately zero mean
        normalized_advs = result["advantages"]
        valid_advs = normalized_advs[result["effective_mask"] > 0]
        assert abs(valid_advs.mean().item()) < 0.1  # Should be close to zero
    
    def test_temperature_scaling(self):
        """Test temperature scaling of logits."""
        config_temp = self.config.copy()
        config_temp["temperature"] = 2.0
        interface_temp = WeightedSFTInterface(config_temp)
        
        result_normal = self.interface.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=self.advantages
        )
        
        result_temp = interface_temp.compute_weighted_loss(
            logits=self.logits,
            tokens=self.tokens,
            loss_masks=self.loss_masks,
            advantages=self.advantages
        )
        
        # Temperature scaling should affect the loss
        assert result_normal["loss"].item() != result_temp["loss"].item()


class TestAtroposBatchProcessor:
    """Test cases for the AtroposBatchProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = AtroposBatchProcessor(pad_token_id=0, max_length=8)
    
    def test_basic_batch_processing(self):
        """Test basic batch processing from Atropos format."""
        # Sample Atropos batch
        atropos_batch = [
            {
                "tokens": [[1, 2, 3], [4, 5]],
                "masks": [[1, 1, 1], [1, 1]],
                "scores": [1.0, 0.5],
                "advantages": [[1.0, 1.5, 2.0], [0.5, 1.0]]
            }
        ]
        
        result = self.processor.process_atropos_batch(atropos_batch)
        
        assert "tokens" in result
        assert "loss_masks" in result
        assert "advantages" in result
        
        # Check padding
        assert len(result["tokens"]) == 2  # Two sequences
        assert len(result["tokens"][0]) == 8  # Padded to max_length
        assert len(result["tokens"][1]) == 8  # Padded to max_length
    
    def test_sequence_level_advantages(self):
        """Test handling of sequence-level advantages."""
        atropos_batch = [
            {
                "tokens": [[1, 2, 3]],
                "masks": [[1, 1, 1]],
                "scores": [2.0],
                # No advantages provided, should use scores
            }
        ]
        
        result = self.processor.process_atropos_batch(atropos_batch)
        
        # Should use scores as sequence-level advantages
        assert result["advantages"][0] == [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def test_tensor_conversion(self):
        """Test conversion to PyTorch tensors."""
        batch = {
            "tokens": [[1, 2, 3, 0], [4, 5, 0, 0]],
            "loss_masks": [[1, 1, 1, 0], [1, 1, 0, 0]],
            "advantages": [[1.0, 1.5, 2.0, 0.0], [0.5, 1.0, 0.0, 0.0]],
            "labels": None
        }
        
        tensors = self.processor.to_tensors(batch, device="cpu")
        
        assert isinstance(tensors["tokens"], torch.Tensor)
        assert isinstance(tensors["loss_masks"], torch.Tensor)
        assert isinstance(tensors["advantages"], torch.Tensor)
        
        assert tensors["tokens"].dtype == torch.long
        assert tensors["loss_masks"].dtype == torch.float
        assert tensors["advantages"].dtype == torch.float


def test_end_to_end_integration():
    """Test end-to-end integration of the components."""
    # Create sample Atropos batch
    atropos_batch = [
        {
            "tokens": [[1, 2, 3, 4], [5, 6, 7]],
            "masks": [[1, 1, 1, 1], [1, 1, 1]],
            "scores": [1.0, 0.5],
            "advantages": [[1.0, 1.5, 2.0, 1.0], [0.5, 1.0, 1.5]]
        }
    ]
    
    # Process batch
    processor = AtroposBatchProcessor(pad_token_id=0, max_length=6)
    processed_batch = processor.process_atropos_batch(atropos_batch)
    tensors = processor.to_tensors(processed_batch, device="cpu")
    
    # Create sample logits
    batch_size, seq_len = tensors["tokens"].shape
    vocab_size = 10
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Compute weighted loss
    interface = WeightedSFTInterface()
    result = interface.compute_weighted_loss(
        logits=logits,
        tokens=tensors["tokens"],
        loss_masks=tensors["loss_masks"],
        advantages=tensors["advantages"]
    )
    
    # Verify result structure
    assert "loss" in result
    assert result["loss"].dim() == 0  # Scalar loss
    assert not torch.isnan(result["loss"])  # Valid loss value
    assert result["loss"].item() >= 0  # Non-negative loss


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
