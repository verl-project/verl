# Atropos Weighted SFT Interface

This document describes the **Loss-Masked Weighted SFT Interface** for Atropos, which enables token-level advantage-weighted supervised fine-tuning integrated with the Atropos RL environment system.

## Overview

The Weighted SFT Interface provides a clean, pluggable solution for:

1. **Token-level Cross-Entropy Loss**: Computes standard cross-entropy loss at the token level
2. **Advantage Weighting**: Scales loss by token-level or sequence-level advantages
3. **Loss Masking**: Selectively includes/excludes tokens from loss computation
4. **Atropos Integration**: Seamlessly processes batches from the Atropos API

This is particularly useful for RLHF-style training where you want to upweight or downweight specific tokens based on their quality or importance.

## Key Components

### 1. WeightedSFTInterface

The core class that handles loss computation:

```python
from atropos_sft_interface import WeightedSFTInterface

# Initialize with configuration
config = {
    "loss_reduction": "mean",        # How to reduce loss: "mean", "sum", "none"
    "ignore_index": -100,           # Token index to ignore (standard for padding)
    "advantage_normalization": "batch",  # Normalize advantages: "batch", "sequence", "none"
    "temperature": 1.0              # Temperature scaling for logits
}

interface = WeightedSFTInterface(config)

# Compute weighted loss
result = interface.compute_weighted_loss(
    logits=model_logits,           # (batch_size, seq_len, vocab_size)
    tokens=input_tokens,           # (batch_size, seq_len)
    loss_masks=loss_masks,         # (batch_size, seq_len) - 1=include, 0=exclude
    advantages=advantages          # (batch_size, seq_len) or (batch_size,)
)

loss = result["loss"]  # Final scalar loss for backprop
```

### 2. AtroposBatchProcessor

Converts Atropos API batches to the required format:

```python
from atropos_sft_interface import AtroposBatchProcessor

processor = AtroposBatchProcessor(pad_token_id=0, max_length=512)

# Process Atropos batch
processed_batch = processor.process_atropos_batch(atropos_batch)
tensors = processor.to_tensors(processed_batch, device="cuda")
```

### 3. WeightedSFTTrainer

Example trainer showing complete integration:

```python
from example_trainer.weighted_sft_trainer import WeightedSFTTrainer, WeightedSFTConfig

config = WeightedSFTConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    training_steps=1000,
    advantage_normalization="batch"
)

trainer = WeightedSFTTrainer(config)
trainer.train()
```

## Usage Examples

### Basic Integration

```python
import torch
from atropos_sft_interface import WeightedSFTInterface

# Your existing training loop
model = load_your_model()
interface = WeightedSFTInterface()

for batch in dataloader:
    # Get data from Atropos or your data source
    tokens = batch["tokens"]          # Token sequences
    loss_masks = batch["loss_masks"]  # Which tokens to train on
    advantages = batch["advantages"]  # Token-level advantages
    
    # Forward pass
    logits = model(tokens)
    
    # Compute weighted loss
    result = interface.compute_weighted_loss(
        logits=logits,
        tokens=tokens,
        loss_masks=loss_masks,
        advantages=advantages
    )
    
    # Backward pass
    loss = result["loss"]
    loss.backward()
    optimizer.step()
```

### Atropos Integration

```python
import requests
from atropos_sft_interface import AtroposBatchProcessor

# Setup
processor = AtroposBatchProcessor(pad_token_id=tokenizer.pad_token_id)

# Get batch from Atropos API
response = requests.get("http://localhost:8000/batch")
atropos_batch = response.json()["batch"]

if atropos_batch:
    # Process into training format
    processed_batch = processor.process_atropos_batch(atropos_batch)
    tensors = processor.to_tensors(processed_batch, device="cuda")
    
    # Train with weighted SFT
    result = interface.compute_weighted_loss(
        logits=model(tensors["tokens"]),
        tokens=tensors["tokens"],
        loss_masks=tensors["loss_masks"],
        advantages=tensors["advantages"]
    )
```

## Advantage Patterns

The system supports various advantage patterns:

### Token-Level Advantages
```python
# Each token has its own advantage
advantages = torch.tensor([
    [1.0, 1.5, 2.0, 0.5],  # First sequence
    [0.8, 1.2, 1.8, 1.0]   # Second sequence
])
```

### Sequence-Level Advantages
```python
# One advantage per sequence (broadcasted to all tokens)
advantages = torch.tensor([1.5, 0.8])  # Will be expanded automatically
```

### Common Patterns
- **Uniform Positive**: All tokens weighted equally and positively
- **End Weighted**: Higher weights for tokens near sequence end
- **Sparse Positive**: Only certain important tokens get positive weight
- **Alternating**: Alternating positive/negative weights for testing

## Configuration Options

### Loss Reduction
- `"mean"`: Average loss over all valid tokens (default)
- `"sum"`: Sum loss over all valid tokens  
- `"none"`: Return per-token losses without reduction

### Advantage Normalization
- `"batch"`: Normalize advantages across the entire batch
- `"sequence"`: Normalize advantages within each sequence
- `"none"`: Use raw advantage values

### Temperature Scaling
- Values > 1.0: Softer probability distributions
- Values < 1.0: Sharper probability distributions
- 1.0: No scaling (default)

## Testing

Run the comprehensive test suite:

```bash
cd atropos
python -m pytest tests/test_weighted_sft_interface.py -v
```

Tests cover:
- Basic loss computation
- Advantage weighting correctness
- Loss masking functionality
- Ignore index handling
- Advantage normalization
- Temperature scaling
- End-to-end integration

## Environment Setup

Use the provided test environment to generate synthetic data:

```python
from environments.weighted_sft_environment import WeightedSFTEnvironment
import asyncio

async def run_test_env():
    env = WeightedSFTEnvironment(
        tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
        advantage_patterns=["increasing", "end_weighted", "sparse_positive"]
    )
    await env.run_environment(num_samples=1000, delay=0.1)

asyncio.run(run_test_env())
```

## Integration with Existing Trainers

The interface is designed to be easily integrated into existing training code:

1. **Replace loss computation**: Swap your existing cross-entropy loss with `compute_weighted_loss()`
2. **Add advantage handling**: Process advantages from your data source
3. **Update data loading**: Use `AtroposBatchProcessor` for Atropos integration

The interface maintains compatibility with standard PyTorch training patterns and requires minimal changes to existing code.

## Performance Considerations

- **Memory**: Token-level advantages require additional memory proportional to sequence length
- **Computation**: Advantage normalization adds minimal overhead
- **Batching**: Larger batches improve advantage normalization stability

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Ensure advantages match token dimensions after label shifting
2. **NaN Losses**: Check for invalid advantage values or empty masks
3. **Memory Issues**: Reduce batch size or sequence length for large models

### Debug Information

The interface returns detailed information for debugging:

```python
result = interface.compute_weighted_loss(...)
print(f"Loss: {result['loss'].item()}")
print(f"Valid tokens: {result['effective_mask'].sum().item()}")
print(f"Avg advantage: {result['advantages'].mean().item()}")
```
