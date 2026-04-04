"""
Atropos Weighted SFT Interface

This module provides a clean interface for loss-masked weighted SFT training
that integrates with the Atropos RL environment system. It handles:

1. Token-level cross-entropy loss computation
2. Advantage-based loss weighting  
3. Loss masking for selective token training
4. Batch processing from Atropos API format

The interface is designed to be plugged into existing training loops with minimal changes.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from atroposlib.type_definitions import WeightedSFTBatch, WeightedSFTConfig


class WeightedSFTInterface:
    """
    Main interface for loss-masked weighted SFT training with Atropos integration.
    
    This class provides the core functionality for computing token-level cross-entropy loss
    scaled by advantages and masked appropriately for selective training.
    """
    
    def __init__(self, config: Optional[WeightedSFTConfig] = None):
        """
        Initialize the weighted SFT interface.
        
        Args:
            config: Configuration for the weighted SFT training. If None, uses defaults.
        """
        self.config = config or {
            "loss_reduction": "mean",
            "ignore_index": -100,
            "advantage_normalization": "batch",
            "temperature": 1.0
        }
    
    def compute_weighted_loss(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        loss_masks: torch.Tensor,
        advantages: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute token-level weighted cross-entropy loss.
        
        Args:
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            tokens: Input token sequences of shape (batch_size, seq_len)
            loss_masks: Loss masks of shape (batch_size, seq_len) where 1 = include, 0 = exclude
            advantages: Token-level advantages of shape (batch_size, seq_len) or (batch_size,)
            labels: Target labels of shape (batch_size, seq_len). If None, derived from tokens.
            
        Returns:
            Dictionary containing:
                - 'loss': Final weighted and reduced loss
                - 'token_losses': Per-token losses before reduction
                - 'weighted_losses': Per-token losses after advantage weighting
                - 'effective_mask': Final mask used for loss computation
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Prepare labels (shift tokens by 1 if not provided)
        if labels is None:
            labels = tokens[:, 1:].contiguous()  # Predict next token
            logits = logits[:, :-1, :].contiguous()  # Remove last logit
            loss_masks = loss_masks[:, 1:].contiguous()  # Adjust mask
            if advantages.dim() == 2 and advantages.shape[1] == seq_len:
                advantages = advantages[:, 1:].contiguous()  # Adjust advantages
        
        # Flatten for cross-entropy computation
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_masks = loss_masks.view(-1)
        
        # Apply temperature scaling
        if self.config["temperature"] != 1.0:
            flat_logits = flat_logits / self.config["temperature"]
        
        # Compute token-level cross-entropy losses
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none',
            ignore_index=self.config["ignore_index"]
        )
        
        # Reshape back to (batch_size, seq_len)
        token_losses = token_losses.view(batch_size, -1)
        
        # Create effective mask (combines loss_mask and ignore_index handling)
        effective_mask = flat_masks.view(batch_size, -1).float()
        ignore_mask = (labels != self.config["ignore_index"]).float()
        effective_mask = effective_mask * ignore_mask
        
        # Handle advantages
        if advantages.dim() == 1:
            # Sequence-level advantages, broadcast to token level
            advantages = advantages.unsqueeze(1).expand(-1, token_losses.shape[1])
        elif advantages.shape != token_losses.shape:
            raise ValueError(f"Advantages shape {advantages.shape} doesn't match token losses shape {token_losses.shape}")
        
        # Normalize advantages if requested
        advantages = self._normalize_advantages(advantages, effective_mask)
        
        # Apply advantage weighting
        weighted_losses = token_losses * advantages * effective_mask
        
        # Reduce loss according to configuration
        if self.config["loss_reduction"] == "mean":
            # Mean over all valid tokens
            total_valid_tokens = effective_mask.sum()
            if total_valid_tokens > 0:
                loss = weighted_losses.sum() / total_valid_tokens
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        elif self.config["loss_reduction"] == "sum":
            loss = weighted_losses.sum()
        else:  # "none"
            loss = weighted_losses
        
        return {
            'loss': loss,
            'token_losses': token_losses,
            'weighted_losses': weighted_losses,
            'effective_mask': effective_mask,
            'advantages': advantages
        }
    
    def _normalize_advantages(self, advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize advantages according to the configuration.
        
        Args:
            advantages: Advantage values to normalize
            mask: Mask indicating valid positions
            
        Returns:
            Normalized advantages
        """
        if self.config["advantage_normalization"] == "none":
            return advantages
        
        # Mask out invalid positions for normalization
        masked_advantages = advantages * mask
        
        if self.config["advantage_normalization"] == "batch":
            # Normalize across the entire batch
            valid_advantages = masked_advantages[mask > 0]
            if len(valid_advantages) > 1:
                mean_adv = valid_advantages.mean()
                std_adv = valid_advantages.std()
                if std_adv > 1e-8:
                    advantages = (advantages - mean_adv) / std_adv
        
        elif self.config["advantage_normalization"] == "sequence":
            # Normalize per sequence
            for i in range(advantages.shape[0]):
                seq_mask = mask[i] > 0
                if seq_mask.sum() > 1:
                    seq_advantages = advantages[i][seq_mask]
                    mean_adv = seq_advantages.mean()
                    std_adv = seq_advantages.std()
                    if std_adv > 1e-8:
                        advantages[i] = (advantages[i] - mean_adv) / std_adv
        
        return advantages


class AtroposBatchProcessor:
    """
    Processes batches from Atropos API into the format required for weighted SFT training.
    """
    
    def __init__(self, pad_token_id: int = 0, max_length: Optional[int] = None):
        """
        Initialize the batch processor.
        
        Args:
            pad_token_id: Token ID to use for padding
            max_length: Maximum sequence length (if None, uses the longest sequence in batch)
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def process_atropos_batch(self, atropos_batch: List[Dict]) -> WeightedSFTBatch:
        """
        Convert an Atropos API batch into WeightedSFTBatch format.
        
        Args:
            atropos_batch: Batch from Atropos API containing tokens, masks, scores, advantages
            
        Returns:
            WeightedSFTBatch ready for training
        """
        all_tokens = []
        all_masks = []
        all_advantages = []
        
        for item in atropos_batch:
            tokens = item["tokens"]
            masks = item["masks"] 
            scores = item.get("scores", [1.0] * len(tokens))
            advantages = item.get("advantages", None)
            
            # Handle advantages - can be per-sequence or per-token
            if advantages is None:
                # Use scores as sequence-level advantages
                item_advantages = [[score] * len(token_seq) for score, token_seq in zip(scores, tokens)]
            else:
                # Use provided token-level advantages
                item_advantages = advantages
            
            all_tokens.extend(tokens)
            all_masks.extend(masks)
            all_advantages.extend(item_advantages)
        
        # Pad sequences to same length
        max_len = self.max_length or max(len(seq) for seq in all_tokens)
        
        padded_tokens = []
        padded_masks = []
        padded_advantages = []
        
        for tokens, masks, advantages in zip(all_tokens, all_masks, all_advantages):
            # Pad tokens
            padded_tokens.append(
                tokens + [self.pad_token_id] * (max_len - len(tokens))
            )
            
            # Pad masks (0 for padded positions)
            padded_masks.append(
                masks + [0] * (max_len - len(masks))
            )
            
            # Pad advantages (0 for padded positions)
            if len(advantages) == 1:
                # Sequence-level advantage, broadcast to all tokens
                advantages = advantages * len(tokens)
            padded_advantages.append(
                advantages + [0.0] * (max_len - len(advantages))
            )
        
        return {
            "tokens": padded_tokens,
            "loss_masks": padded_masks,
            "advantages": padded_advantages,
            "labels": None  # Will be derived from tokens
        }
    
    def to_tensors(self, batch: WeightedSFTBatch, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Convert WeightedSFTBatch to PyTorch tensors.
        
        Args:
            batch: WeightedSFTBatch to convert
            device: Device to place tensors on
            
        Returns:
            Dictionary of tensors ready for training
        """
        return {
            "tokens": torch.tensor(batch["tokens"], dtype=torch.long, device=device),
            "loss_masks": torch.tensor(batch["loss_masks"], dtype=torch.float, device=device),
            "advantages": torch.tensor(batch["advantages"], dtype=torch.float, device=device),
            "labels": torch.tensor(batch["labels"], dtype=torch.long, device=device) if batch["labels"] is not None else None
        }
