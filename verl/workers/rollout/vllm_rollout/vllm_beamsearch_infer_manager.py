import ray

@ray.remote(num_gpus=1)
class VLLMBeamSearchInfer:
    def __init__(self, model_path, sampling_params, cuda_device:int):
        import os
        import tempfile
        # Ray automatically sets CUDA_VISIBLE_DEVICES when num_gpus=1
        # But we still verify it's set correctly
        print(f"Actor {cuda_device}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        
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
        self.llm = LLM(
            model=model_path, 
            tokenizer=model_path, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.6
        )
        self.sampling_params = sampling_params
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, batch:list[dict]) -> list[str]:
        # vLLM's generate expects string prompts, not token IDs
        if (len(batch) == 0):
            return []
        batch = [{"prompt": self.tokenizer.apply_chat_template(prompt["prompt"], tokenize=False, add_generation_prompt=True)} for prompt in batch]
        # print(f"Heartbeat: Before beam search, batch size: {len(batch)}")
        outputs = self.llm.beam_search(batch, params=self.sampling_params)
        return [outputs[i].sequences for i in range(len(outputs))]

    def sleep(self):
        self.llm.sleep(level=1)
    
    def wake_up(self):
        self.llm.wake_up()
        self.llm.reset_prefix_cache()
    
class VLLMBeamSearchManager:
    def __init__(self, model_path, config, num_gpus:int):
        self.model_path = model_path
        self.config = config
        self.beam_search_config = config.get("beam_search_config", {})
        
        from vllm.sampling_params import BeamSearchParams
        self.sampling_params = BeamSearchParams(beam_width=self.beam_search_config.get("beam_width", 1), max_tokens=self.beam_search_config.get("max_tokens", 1024))
        
        # default per gpu one worker
        self.num_gpus = num_gpus
        self.workers = [VLLMBeamSearchInfer.remote(model_path, self.sampling_params, i) for i in range(num_gpus)]

        
    def generate(self, batch:list[dict]) -> list[str]:
        bsz = len(batch)//self.num_gpus
        remain = len(batch)%self.num_gpus
        
        batches = []
        
        start = 0
        for i in range(self.num_gpus):
            end = start + bsz
            if i < remain:
                end += 1
            batches.append({batch[start:end]})
            start = end
        
        results = ray.get([worker.generate.remote(batch) for worker, batch in zip(self.workers, batches)])
        return results

    def sleep(self):
        ray.get([worker.sleep.remote() for worker in self.workers])
    
    def wake_up(self):
        ray.get([worker.wake_up.remote() for worker in self.workers])
    
    def __call__(self, batch:list[dict]) -> list[str]:
        return self.generate(batch)
    