
Add models with the FSDP backend
==================================

Last updated: 02/09/2025.

Model
--------------------------

In principle, our FSDP backend can support any HF model and we can
sychronoize the actor model weight with vLLM using `hf_weight_loader.py` under `third_party/vllm`.
However, ``hf_weight_loader`` is will gather the full state_dict of a
model during synchronization, which may cause OOM. We suggest using
``dtensor_weight_loader`` which gather the full model parameter layer by
layer to reduce the peak memory usage. We already support dtensor weight
loader for the models below in `dtensor_weight_loader.py` under `third_party/vllm`:

- ``GPT2LMHeadModel``
- ``LlamaForCausalLM``
- ``LLaMAForCausalLM``
- ``MistralForCausalLM``
- ``InternLMForCausalLM``
- ``AquilaModel``
- ``AquilaForCausalLM``
- ``Phi3ForCausalLM``
- ``GemmaForCausalLM``
- ``Gemma2ForCausalLM``
- ``GPTBigCodeForCausalLM``
- ``Starcoder2ForCausalLM``
- ``Qwen2ForCausalLM``
- ``DeepseekV2ForCausalLM``

To implement ``dtensor_weight_loader`` of a model that's supported in
vLLM, follow the guide of gemma model below:

1. Copy the
   ``load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])`` from the vllm model class
   to ``dtensor_weight_loaders.py``
2. Modify the arguments to
   ``(actor_weights: Dict, vllm_model: nn.Module)``
3. Replace the ``self`` to ``vllm_model``
4. Add the
   ``local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)``
   before each ``param = params_dict[name]`` and modify the following
   weight loading using ``local_loaded_weight``.
5. Register the implemented dtensor weight loader to ``__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__``.

.. code-block:: diff

    - def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    + def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
    -   params_dict = dict(self.named_parameters())
    +   params_dict = dict(vllm_model.named_parameters())
        loaded_params = set()
    -   for name, loaded_weight in weights:
    +   for name, loaded_weight in actor_weights.items():
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
    +           local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                param = params_dict[name]
                weight_loader = param.weight_loader
    -           weight_loader(param, loaded_weight, shard_id)
    +           weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
    +           local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
    -           weight_loader(param, loaded_weight)
    +           weight_loader(param, local_loaded_weight.to(dtype=param.dtype))
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}")