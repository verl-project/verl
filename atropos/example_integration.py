"""
Simple Integration Example for Atropos Weighted SFT Interface

This example shows how to integrate the weighted SFT interface into
an existing training loop with minimal changes.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import time

from atropos_sft_interface import WeightedSFTInterface, AtroposBatchProcessor


def original_training_step(model, tokens, labels, mask):
    """
    Original training step using standard cross-entropy loss.
    This is what you might have before integrating Atropos.
    """
    # Forward pass
    outputs = model(tokens)
    logits = outputs.logits
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    
    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)
    
    token_losses = loss_fct(flat_logits, flat_labels)
    token_losses = token_losses.view(shift_labels.shape)
    
    # Apply mask and reduce
    masked_losses = token_losses * shift_mask
    loss = masked_losses.sum() / shift_mask.sum()
    
    return loss


def weighted_sft_training_step(model, tokens, labels, mask, advantages, interface):
    """
    Updated training step using the weighted SFT interface.
    This shows the minimal changes needed to integrate Atropos.
    """
    # Forward pass (unchanged)
    outputs = model(tokens)
    logits = outputs.logits
    
    # Use weighted SFT interface instead of manual loss computation
    result = interface.compute_weighted_loss(
        logits=logits,
        tokens=tokens,
        loss_masks=mask,
        advantages=advantages,
        labels=labels
    )
    
    return result["loss"], result


def simple_atropos_integration():
    """
    Simple example showing how to integrate with Atropos API.
    """
    # Setup model and tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    
    # Setup weighted SFT interface
    interface = WeightedSFTInterface({
        "loss_reduction": "mean",
        "ignore_index": -100,
        "advantage_normalization": "batch",
        "temperature": 1.0
    })
    
    # Setup batch processor for Atropos data
    processor = AtroposBatchProcessor(
        pad_token_id=tokenizer.pad_token_id,
        max_length=512
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    print("Starting training with Atropos weighted SFT...")
    
    # Training loop
    for step in range(100):
        try:
            # Get batch from Atropos API
            response = requests.get("http://localhost:8000/batch", timeout=5)
            data = response.json()
            atropos_batch = data.get("batch")
            
            if atropos_batch is None:
                print(f"Step {step}: No batch available, waiting...")
                time.sleep(1)
                continue
            
            # Process Atropos batch
            processed_batch = processor.process_atropos_batch(atropos_batch)
            tensors = processor.to_tensors(processed_batch, device=device)
            
            # Training step with weighted SFT
            loss, result = weighted_sft_training_step(
                model=model,
                tokens=tensors["tokens"],
                labels=tensors["labels"],
                mask=tensors["loss_masks"],
                advantages=tensors["advantages"],
                interface=interface
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            if step % 10 == 0:
                with torch.no_grad():
                    num_tokens = result["effective_mask"].sum().item()
                    avg_advantage = result["advantages"][result["effective_mask"] > 0].mean().item()
                
                print(f"Step {step}:")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Tokens: {num_tokens}")
                print(f"  Avg Advantage: {avg_advantage:.3f}")
        
        except requests.RequestException:
            print(f"Step {step}: Failed to get batch from Atropos API")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"Step {step}: Error - {e}")
            continue
    
    print("Training completed!")


def comparison_example():
    """
    Example comparing original vs weighted SFT training.
    """
    # Create sample data
    batch_size, seq_len, vocab_size = 2, 8, 1000
    
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = tokens.clone()
    mask = torch.ones_like(tokens, dtype=torch.float)
    advantages = torch.tensor([[1.0, 1.5, 2.0, 1.0, 0.5, 1.2, 1.8, 1.0],
                              [0.8, 1.2, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4]])
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, 128),
        torch.nn.Linear(128, vocab_size)
    )
    
    # Mock logits
    with torch.no_grad():
        embeddings = model[0](tokens)
        logits = model[1](embeddings)
    
    print("=== Comparison: Original vs Weighted SFT ===")
    
    # Original loss computation
    original_loss = original_training_step(
        model=lambda x: type('obj', (object,), {'logits': logits})(),
        tokens=tokens,
        labels=labels,
        mask=mask
    )
    print(f"Original Loss: {original_loss.item():.4f}")
    
    # Weighted SFT loss computation
    interface = WeightedSFTInterface()
    result = interface.compute_weighted_loss(
        logits=logits,
        tokens=tokens,
        loss_masks=mask,
        advantages=advantages
    )
    print(f"Weighted SFT Loss: {result['loss'].item():.4f}")
    
    # Show the effect of advantages
    print(f"Average advantage: {advantages.mean().item():.3f}")
    print(f"Advantage std: {advantages.std().item():.3f}")
    
    # Test with uniform advantages (should be similar to original)
    uniform_advantages = torch.ones_like(advantages)
    uniform_result = interface.compute_weighted_loss(
        logits=logits,
        tokens=tokens,
        loss_masks=mask,
        advantages=uniform_advantages
    )
    print(f"Uniform Advantage Loss: {uniform_result['loss'].item():.4f}")


def minimal_integration_example():
    """
    Minimal example showing the smallest possible integration.
    """
    print("=== Minimal Integration Example ===")
    
    # Your existing training code might look like this:
    def your_existing_loss_function(logits, labels, mask):
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1)
        
        losses = loss_fct(flat_logits, flat_labels).view(shift_labels.shape)
        return (losses * shift_mask).sum() / shift_mask.sum()
    
    # To integrate weighted SFT, just replace with:
    def your_new_loss_function(logits, labels, mask, advantages):
        interface = WeightedSFTInterface()
        result = interface.compute_weighted_loss(
            logits=logits,
            tokens=labels,  # Use labels as tokens for this example
            loss_masks=mask,
            advantages=advantages
        )
        return result["loss"]
    
    # Example usage
    batch_size, seq_len, vocab_size = 2, 6, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones_like(labels, dtype=torch.float)
    advantages = torch.ones_like(labels, dtype=torch.float) * 1.5  # 50% higher weight
    
    old_loss = your_existing_loss_function(logits, labels, mask)
    new_loss = your_new_loss_function(logits, labels, mask, advantages)
    
    print(f"Old loss function: {old_loss.item():.4f}")
    print(f"New loss function: {new_loss.item():.4f}")
    print(f"Ratio (new/old): {(new_loss/old_loss).item():.3f}")


if __name__ == "__main__":
    print("Atropos Weighted SFT Integration Examples")
    print("=" * 50)
    
    # Run comparison example
    comparison_example()
    print()
    
    # Run minimal integration example
    minimal_integration_example()
    print()
    
    # Uncomment to run full Atropos integration (requires running Atropos API)
    # simple_atropos_integration()
    
    print("Examples completed!")
    print("\nTo run the full Atropos integration:")
    print("1. Start the Atropos API server")
    print("2. Start an environment (e.g., weighted_sft_environment.py)")
    print("3. Uncomment and run simple_atropos_integration()")
