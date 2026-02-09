updated 20251222

# The ways verl integrates megatron-core
There has been 3 ways that verl integrates megatron-core as it training backend:
1. the codes inside this directory, which defines the conversion for new models one by one. (deprecated now)
2. through [mbridge](https://github.com/ISEEKYAN/mbridge) (will be deprecated at about v0.8)
3. through [megatron-bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) (the official way for further development)

There is a configure option of `megatron.use_mbridge` to choose way#1 (false) or way#2 (true), and after the megatron-bridge is integrated we have a new option `megatron.vanilla_mbridge` to choose way#2 (true) or way#3 (false)

Now since we deprecated the way#1, the option `use_mbridge` will be asserted to be true and will be removed after v0.7. The default `vanilla_mbridge` is true for now and will be false one the megatron-bridge backend turns default.

With the bridge way(#2 or #3), we can directly load and save the megatron model weight through HuggingFace format, and we can use any megatron version >= 0.13 to adopt new megatron optimization feature as handy as possible by directly add overrided megatron configs such as `+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform`.

# How to support new models
1. Make sure the model is supported by your inference engine (vLLM or SGLang or TensorRT-LLM) with correct version.
2. Make sure the model is supported by the bridge
   - If it is a model of new architecture, open an issue to `megatron-bridge` or contribute your implementation to `megatron-bridge`. Be cautious to have a matched version of `Megatron` and `TransformerEngine`
   - If it is a private model, implement your private model with `mbridge` or `megatron-bridge`(prefered).

3. Now the model is supported, just change the model path of your scripts and run the scritps.





# #Below are deprecated since 2025.12#
# verl Megatron-Core Models
Now we use [mbridge](https://github.com/iseekyan/mbridge) to support megatron models. And we will migrate to [megatron-bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) in the future.

With the mbridge, we can use allmost all the Megatron-Core features to support new models with little effort. And no offline weights conversion is needed, all the weights conversion is done online. We can directly save the mcore model to huggingface format during training.

Also, we can easily upgrade the mcore version to the latest version. In most cases, the upgrade is seamless. (except when the mcore API changes and we need to update the verl code accordingly)

## How to support new models
1. make sure the model is supported by vLLM
2. Support the model in [mbridge](https://github.com/iseekyan/mbridge), see its currently supported models for example.
    - we will migrate to [megatron-bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) in the future.
3. Register the model forward function in verl, see the example in `verl/verl/models/mcore/registry.py`.



# #Below are deprecated since 2025.10#
The earlier versions of verl use `Megatron-LM` 0.4 and workaround huggingface model classes. To better use the latest features and speedup of modern Megatron, we are migrating to `Megatron-Core`(mcore), and use the recommended `GPTModel` class for all language models. With mcore `GPTModel`, we can use the latest features like `context parallel`, `expert parallel`, `dist_checkpointing`, etc. and we can update mcore with little effort in the future for new features.

The migration has been successful with the help of the mcore team and the community. What we have done is:
1. update `Megatron` version to `0.14.0`
2. migrate `LlamaForCausalLM` and `Qwen2ForCausalLM` to mcore `GPTModel`
3. support sequence packing/thd format.
4. support `tensor parallel`, `pipeline parallel`, `sequence parallel`, `virtual pipeline parallel`, `context parallel`.
5. support the mcore `dist_checkpointing` feature and a basic offline weighs conversion script from huggingface to mcore `dist_checkpointing` format.

We are working on the following features:
- support `Qwen2MoeForCausalLM`
- support `MixtralForCausalLM`
- support `DeepseekV3ForCausalLM`
- support `expert parallel`

Features we invite the community to contribute:
- better scripts for offline weights conversion from huggingface to mcore `dist_checkpointing` format.
    - conversion of large models with multiple GPUs
    - conversion of large models with single GPU
- refactor the `megatron_checkpoint_manager.py` by `dist_checkpointing` format.
- support llama4
- support qwen2.5-vl

To track the progress of verl mcore integration, please refer to the [mcore integration issue](https://github.com/volcengine/verl/issues/1033).

## How things work now
To engage the community in contributing, here are the key steps in our mcore integration process and features under development. 

The huggingface `transformers` is the de facto standard of model zoo while mcore is good at computation efficiency. The main challenge is conversion between the two.
main steps:
1. modelling the huggingface model with mcore `GPTModel`
    - a. convert the huggingface config to mcore `TransformerConfig`
    - b. init the mcore `GPTModel` with the converted config
    - c. load the huggingface model weights to the `GPTModel`
2. online weight conversion from mcore to huggingface (due to the rollout engine `vLLM` is using huggingface format)
    - a. bridge the gap between mcore and huggingface weights format and name mapping
    - b. online resharding the mcore weights to rollout engine
        - this part is very complicated with multiple parallel strategies composition between mcore and rollout engine
3. support the mcore features in verl
    - a. support `tensor parallel`, `pipeline parallel`, `sequence parallel`, `virtual pipeline parallel`, `context parallel`
    - b. support recompute and other mcore speed up features

4. checkpointing
    - a. support recovering the verl training.
    - b. support exporting the mcore checkpoint to huggingface format, for downstream inference.

### Modelling the huggingface model with mcore `GPTModel`
The first step is to convert huggingface config to mcore `TransformerConfig` and init the mcore `GPTModel` with the converted config. See code in `verl/models/mcore/config_converter.py` and `verl/verl/models/mcore/models/model_initializer.py`. The corresponding model forward code is in `verl/verl/models/mcore/models/model_forward.py`.

There are two ways of loading the huggingface model weights to the `GPTModel`
1. Runtime loading
    - every rank loads the entire huggingface model weights and then shard and convert to mcore weights.
    - speed is slow and memory consumption is high.
    - this way is deprecated and will not support new models.
2. Offline loading
    - use offline script to convert the huggingface model weights to mcore weights and save with mcore `dist_checkpointing` format.
    - online loading and sharding is automatically done by mcore `dist_checkpointing` format. The speed is fast and memory consumption is low.
    - the offline script is in `verl/scripts/converter_hf_to_mcore.py`.

### online weight conversion from mcore to huggingface
See function `convert_megatron_model_to_transformers_model` in `verl/utils/megatron_utils.py` for the details.

It should be refatored for extensibility and better performance.

### support the mcore features in verl
Most of the features of `GPTModel` is out-of-the-box supported in verl through changing the `TransformerConfig`, except those about parallel strategies, such as `expert parallel`. 
Features about parallel strategies should be supported with changes about the online weights conversion(especially the resharding part) and verl work dispatching.

### checkpointing
The existing checkpointing code is in `verl/utils/checkpoint/megatron_checkpoint_manager.py`. And the script to convert checkpoint to huggingface format is in `verl/scripts/model_merger`.

The existing checkpoint format simply saves every rank's weights and optimizer states. It should be refactored by `dist_checkpointing` format.


## How to support new models
1. make sure the model is supported by vLLM
2. modelling the huggingface model with mcore `GPTModel` (The [Pai-Megatron-Path](https://github.com/alibaba/Pai-Megatron-Patch/tree/main) is a good reference)
    - a. convert the huggingface config to mcore `TransformerConfig`
    - b. init the mcore `GPTModel` with the converted config
    - c. load the huggingface model weights to the `GPTModel`
    - d. for VLM the interface might be different, it is ok to add a new model class with GPTModel as its module.
3. offline weights conversion from huggingface to mcore `dist_checkpointing` format
4. support online weights conversion from mcore to huggingface
    - it is recommended to initialize a vLLM model with the converted mcore weights, and then test if the generating sequence is correct.


## How to scale up to larger models like deepseek-v3 or other 100B+ models
The greatest challenge for scaling up to larger models is the memory consumption.

The necessary features under development for scaling up are
1. Training engine part
    - expert parallel
2. Rollout engine part
    - pipeline parallel
    - expert parallel
    - more efficient and general weight resharding and loading
3. Offline weights conversion
    - support weights larger than single GPU memory
