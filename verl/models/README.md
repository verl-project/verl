# Models
Common modelzoo such as huggingface/transformers stuggles when using Pytorch native model parallelism. Following the design principle of vLLM, we keep a simple, parallelizable, highly-optimized with packed inputs in verl. 
## Adding a New Huggingface Model
### Step 1: Copy the model file from HF to verl
- Add a new file under verl/models/hf
- Copy ONLY the model file from huggingface/transformers/models to verl/models/hf

### Step 2: Modify the model file to use packed inputs
- Remove all the code related to inference (kv cache)
- Modify the inputs to include only
    - input_ids (total_nnz,)
    - cu_seqlens (total_nnz + 1,)
    - max_seqlen_in_batch: int
- Note that this requires using flash attention with causal mask.

### Step 2.5: Add tests
- Add a test to compare this version and the huggingface version
- Following the infrastructure and add tests to tests/models/hf

### Step 3: Add a function to apply tensor parallelism
- Please follow
    - https://pytorch.org/docs/stable/distributed.tensor.parallel.html
    - https://pytorch.org/tutorials/intermediate/TP_tutorial.html
- General comments
    - Tensor Parallelism in native Pytorch is NOT auto-parallelism. The way it works is to specify how model parameters and input/output reshards using configs. These configs are then registered as hooks to perform input/output resharding before/after model forward.

### Step 4: Add a function to apply data parallelism
- Please use FSDP2 APIs
- See demo here https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L413

### Step 5: Add a function to apply pipeline parallelism
- Comes in Pytorch 2.4
- Currently only in alpha in nightly version
- Check torchtitan for more details

