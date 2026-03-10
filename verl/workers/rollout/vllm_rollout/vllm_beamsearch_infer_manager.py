import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import logging
import asyncio

logger = logging.getLogger(__name__)

@ray.remote
class VLLMBeamSearchInfer:
    """A single vLLM beam search inference actor.

    GPU resource allocation:
    - With placement group: num_gpus=0.25 (fractional) + bundle binding guarantees
      each actor is on a different GPU while allowing co-location with FusedWorker.
    - Without placement group: num_gpus=1 guarantees each actor gets its own GPU.
    The num_gpus is set dynamically in VLLMBeamSearchManager, NOT via decorator default.
    """

    def __init__(self, model_path, sampling_params, cuda_device:int, memory_utilization:float, max_model_len:int=4096, dtype:str="bfloat16"):
        import os
        import tempfile
        
        # Log actor initialization details
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
        logger.info(
            f"VLLMBeamSearchInfer actor {cuda_device} initializing: "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"memory_utilization={memory_utilization}"
        )
        
        # Ray sets CUDA_VISIBLE_DEVICES based on num_gpus allocation
        print(f"Actor {cuda_device}: CUDA_VISIBLE_DEVICES={cuda_visible}")
        
        # Set independent cache directories for each actor to avoid torch.compile cache conflicts
        # vLLM uses ~/.cache/vllm/torch_compile_cache/ by default
        # We set XDG_CACHE_HOME to use a per-actor cache directory
        actor_cache_home = os.path.join(tempfile.gettempdir(), f"vllm_cache_actor_{cuda_device}")
        os.makedirs(actor_cache_home, exist_ok=True)
        
        # Set XDG_CACHE_HOME so vLLM will use actor-specific cache directory
        # This ensures each actor has its own ~/.cache equivalent
        os.environ["XDG_CACHE_HOME"] = actor_cache_home
        
        # Also set torch.compile cache directory as a fallback
        torch_compile_cache = os.path.join(actor_cache_home, "vllm", "torch_compile_cache")
        os.makedirs(torch_compile_cache, exist_ok=True)
        os.environ["TORCH_COMPILE_CACHE_DIR"] = torch_compile_cache
        
        # Import vLLM inside __init__ so that CUDA_VISIBLE_DEVICES env var is already set
        from vllm import LLM
        logger.info(f"VLLMBeamSearchInfer actor {cuda_device} starting to load vLLM model: {model_path}")
        
        # Configure vLLM with minimal memory footprint for shared GPU scenario
        # disable_log_stats=True to reduce memory overhead
        # enable_prefix_caching=False to disable prefix caching optimization (KV cache itself is mandatory)
        # max_num_seqs limits concurrent requests to reduce KV cache memory
        # max_num_batched_tokens limits batch processing to reduce peak memory
        self.llm = LLM(
            model=model_path, 
            tokenizer=model_path, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=memory_utilization,
            enable_prefix_caching=True,  # Disable prefix caching optimization
            disable_log_stats=True,  # Reduce memory overhead
            max_model_len=max_model_len,  # Configurable max sequence length
            # max_num_seqs=128,  # Limit concurrent sequences to reduce KV cache memory
            max_num_batched_tokens=max_model_len,  # Match max_model_len to avoid conflicts
            # enforce_eager=True,  # Disable CUDA graph to save memory
            dtype=dtype,
        )
        self.sampling_params = sampling_params
        self.tokenizer = self.llm.get_tokenizer()
        logger.info(
            f"VLLMBeamSearchInfer actor {cuda_device} successfully loaded vLLM model, "
            f"GPU memory utilization target: {memory_utilization}, KV cache disabled"
        )

    def ready(self) -> bool:
        """Return True when actor init (including vLLM load) is complete. Used to serialize init and avoid GPU memory spike."""
        return True

    def generate(self, batch:list[dict]) -> list[str]:
        # vLLM's generate expects string prompts, not token IDs
        if (len(batch) == 0):
            return []
        # Store original indices before modifying batch
        original_indices = [item[0] for item in batch]
        # item[1] 是 messages(list[{"role": ..., "content": ...}])
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                item[1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        # vLLM v1 的 beam_search 期望的是形如 {"prompt": str, ...} 的 dict 列表
        prompt_dicts = [{"prompt": text} for text in prompt_texts]
        outputs = self.llm.beam_search(prompt_dicts, params=self.sampling_params)
        # Extract text from BeamSearchSequence objects and return (original_index, text_list) pairs
        # outputs[i].sequences is a list of BeamSearchSequence objects, each has .text attribute
        return [(original_indices[i], [seq.text for seq in outputs[i].sequences]) for i in range(len(outputs))]

    def sleep(self):
        self.llm.sleep(level=1)
    
    def wake_up(self):
        self.llm.wake_up()
        self.llm.reset_prefix_cache()
    
class VLLMBeamSearchManager:
    def __init__(
        self,
        model_path,
        config,
        num_gpus: int,
        placement_group=None,
        start_bundle_index: int = None,
    ):
        """
        Args:
            model_path: Path to the model.
            config: Reward config (beam_search_config etc).
            num_gpus: Number of GPU workers (VLLMBeamSearchInfer actors).
            placement_group: If set, place workers in this placement group (shared global_pool).
            start_bundle_index: First bundle index for this manager in the placement group.
        """
        self.model_path = model_path
        self.config = config
        self.beam_search_config = config.get("beam_search_config", {})
        self.dtype = self.beam_search_config.get("dtype", "bfloat16")
        
        from vllm.sampling_params import BeamSearchParams
        self.sampling_params = BeamSearchParams(
            beam_width=self.beam_search_config.get("beam_width", 1),
            max_tokens=self.beam_search_config.get("max_tokens", 1024),
        )
        tp_size = self.beam_search_config.get("tensor_model_parallel_size", 1)
        memory_utilization = self.beam_search_config.get("gpu_memory_utilization", 0.5)
        # max_model_len for vLLM: prompt + output tokens must fit within this limit
        # Default 4096 to accommodate longer prompts from actor model outputs
        self.max_model_len = self.beam_search_config.get("max_model_len", 4096)
        
        
        
        self.num_gpus = num_gpus
        
        # Log placement group and bundle information before creating workers
        if placement_group is not None and start_bundle_index is not None:
            logger.info(
                f"VLLMBeamSearchManager initializing with placement_group, "
                f"start_bundle_index={start_bundle_index}, num_gpus={num_gpus}"
            )
            
            # Check available GPU resources in the placement group
            try:
                pg_state = ray.util.placement_group_table(placement_group)
                logger.info(f"Placement group state: {pg_state.get('state', 'unknown')}")
                logger.info(f"Placement group bundles: {placement_group.bundle_count}")
                
                # Check available resources before creating actors
                available_resources = ray.available_resources()
                logger.info(
                    f"Cluster available resources before VLLMBeamSearchInfer creation: "
                    f"GPU={available_resources.get('GPU', 0):.2f}, "
                    f"CPU={available_resources.get('CPU', 0):.2f}"
                )
            except Exception as e:
                logger.warning(f"Failed to query placement group state: {e}")
            
            # Share global_pool: schedule each infer actor in a DIFFERENT bundle of the placement group.
            # Each bundle corresponds to one GPU. Using num_gpus=0.25 (fractional) allows co-location
            # with FusedWorker (actor/rollout/ref) on the same GPU, while explicit bundle_index binding
            # guarantees each VLLMBeamSearchInfer is on a different GPU.
            self.workers = []
            for i in range(num_gpus):
                bundle_idx = start_bundle_index + i
                logger.info(
                    f"Creating VLLMBeamSearchInfer worker {i} on bundle_index={bundle_idx} "
                    f"with num_gpus=0.25 (fractional, co-locate with FusedWorker)"
                )
                try:
                    worker = VLLMBeamSearchInfer.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_bundle_index=bundle_idx,
                        ),
                        num_gpus=0.25,  # Fractional: co-locate with FusedWorker; bundle binding ensures 1 per GPU
                    ).remote(model_path, self.sampling_params, i, memory_utilization, self.max_model_len, self.dtype)
                    self.workers.append(worker)
                    logger.info(f"Successfully created VLLMBeamSearchInfer worker {i} on bundle {bundle_idx}")
                except Exception as e:
                    logger.error(f"Failed to create VLLMBeamSearchInfer worker {i}: {e}")
                    raise
            
            # Check resources after creating actors
            try:
                available_resources_after = ray.available_resources()
                logger.info(
                    f"Cluster available resources after VLLMBeamSearchInfer creation: "
                    f"GPU={available_resources_after.get('GPU', 0):.2f}, "
                    f"CPU={available_resources_after.get('CPU', 0):.2f}"
                )
            except Exception as e:
                logger.warning(f"Failed to query available resources after creation: {e}")
        else:
            # No placement group: use num_gpus=1 to guarantee each worker gets a DIFFERENT GPU.
            # This prevents Ray from scheduling multiple VLLMBeamSearchInfer actors on the same GPU.
            # Note: with num_gpus=1, co-location with FusedWorker is not possible; this fallback
            # is mainly for standalone/testing scenarios. For production with shared GPU (sleep/wake),
            # always use placement group.
            logger.warning(
                f"VLLMBeamSearchManager initializing WITHOUT placement_group. "
                f"Using num_gpus=1 per worker to guarantee 1 worker per GPU. "
                f"For shared GPU with sleep/wake, configure resource pool with placement group."
            )
            self.workers = [
                VLLMBeamSearchInfer.options(num_gpus=1).remote(
                    model_path, self.sampling_params, i, memory_utilization, self.max_model_len, self.dtype
                )
                for i in range(num_gpus)
            ]

        # 注意：不在 __init__ 里立刻调用 sleep()
        # 在共享 GPU 场景下，vLLM 的 sleep 会做一次显存 profile，
        # 如果此时有其他进程释放显存，容易触发
        # 「Memory usage increased after sleeping / Error in memory profiling」断言。
        # 这里让 worker 保持 active，真正用完一次 forward 之后，
        # 由上层 reward_manager (forward_rdkit.run_batch_forward) 显式调用 sleep()。
        logger.info(
            f"VLLMBeamSearchManager created {len(self.workers)} workers; "
            f"workers will be slept only after first forward by caller."
        )

    def wait_ready(self) -> None:
        """Block until all VLLMBeamSearchInfer workers have finished __init__ (vLLM loaded). Use before starting other GPU-heavy init to avoid memory spike."""
        ray.get([w.ready.remote() for w in self.workers])

        
    def generate(self, batch:list[dict]) -> list[str]:
        """Synchronous generate - use generate_async in async contexts to avoid blocking."""
        bsz = len(batch)//self.num_gpus
        remain = len(batch)%self.num_gpus
        
        batches = []
        
        start = 0
        for i in range(self.num_gpus):
            end = start + bsz
            if i < remain:
                end += 1
            batches.append(batch[start:end])
            start = end
        
        results = ray.get([worker.generate.remote(batch) for worker, batch in zip(self.workers, batches)])
        # Flatten results from all workers
        flattened_results = []
        for worker_results in results:
            flattened_results.extend(worker_results)
        return flattened_results

    async def generate_async(self, batch:list[dict]) -> list[str]:
        """Async generate - use this in async contexts to avoid blocking the event loop."""
        bsz = len(batch)//self.num_gpus
        remain = len(batch)%self.num_gpus
        
        batches = []
        
        start = 0
        for i in range(self.num_gpus):
            end = start + bsz
            if i < remain:
                end += 1
            batches.append(batch[start:end])
            start = end
        
        # Use asyncio.gather with ray object refs instead of blocking ray.get
        object_refs = [worker.generate.remote(batch) for worker, batch in zip(self.workers, batches)]
        results = await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])
        # Flatten results from all workers
        flattened_results = []
        for worker_results in results:
            flattened_results.extend(worker_results)
        return flattened_results

    def sleep(self):
        """Synchronous sleep - use sleep_async in async contexts."""
        ray.get([worker.sleep.remote() for worker in self.workers])
    
    async def sleep_async(self):
        """Async sleep - use this in async contexts to avoid blocking the event loop."""
        object_refs = [worker.sleep.remote() for worker in self.workers]
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])
    
    def wake_up(self):
        """Synchronous wake_up - use wake_up_async in async contexts."""
        ray.get([worker.wake_up.remote() for worker in self.workers])
    
    async def wake_up_async(self):
        """Async wake_up - use this in async contexts to avoid blocking the event loop."""
        object_refs = [worker.wake_up.remote() for worker in self.workers]
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])
    
    def __call__(self, batch:list[dict]) -> list[str]:
        return self.generate(batch)
    