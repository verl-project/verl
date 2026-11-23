import json, hashlib, shutil, datetime, sys, os
from pathlib import Path
w=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
run=sys.argv[1]
assert run.startswith('kimi31_4layer_alignment_423_') and '/' not in run
root=w/'results'/run
assert (root/'run-token.txt').exists()
files=[
 'source/vllm/vllm/v1/engine/core.py',
 'source/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py',
 'source/vllm-ascend/vllm_ascend/ascend_forward_context.py',
 'source/alignment_runtime/serving_invariance_probe.py',
 'source/verl/verl/workers/rollout/llm_server.py',
 'source/verl/verl/experimental/agent_loop/single_turn_agent_loop.py',
 'source/alignment_runtime/dsa_alignment.py',
 'model/dsa.py',
 'source/FSDPTurbo/fsdp_turbo/ops/sparse_mla.py',
 'source/FSDPTurbo/fsdp_turbo/ops/sparse_attention.py',
 'source/alignment_runtime/dsa_trace.py',
 'source/vllm-ascend/vllm_ascend/attention/sfa_v1.py',
 'source/alignment_runtime/moe_alignment.py',
 'source/vllm-ascend/vllm_ascend/ops/fused_moe/gate_linear.py',
 'source/vllm-ascend/vllm_ascend/ops/fused_moe/shared_experts.py',
 'source/alignment_runtime/moe_trace.py',
 'source/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py',
 'source/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py',
 'source/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py',
 'source/alignment_runtime/residual_alignment.py',
 'source/alignment_runtime/linear_alignment.py',
 'source/vllm-ascend/vllm_ascend/ops/linear_op.py',
 'source/vllm-ascend/vllm_ascend/ops/mm_encoder_attention.py',
 'source/alignment_runtime/package_metadata_cache.py',
 'source/FSDPTurbo/fsdp_turbo/models/kimi/kimi_k31/dsa.py',
 'source/FSDPTurbo/fsdp_turbo/models/kimi/kimi_k31/modeling_kimi_k31.py',
 'source/FSDPTurbo/fsdp_turbo/models/kimi/kimi_k31/modeling_kimi_linear.py',
 'source/FSDPTurbo/fsdp_turbo/models/kimi/kimi_k31/moe.py',
 'source/FSDPTurbo/fsdp_turbo/ops/npu/flash_attn.py',
 'source/FSDPTurbo/fsdp_turbo/ops/rope.py',
 'source/vllm-ascend/vllm_ascend/worker/worker.py',
 'source/vllm/vllm/model_executor/models/kimi_k25_vit.py',
 'model/modeling_kimi_k31.py',
 'results/cann-view-provenance.json',
 'source/alignment_runtime/mm_trace.py','source/vllm-ascend/vllm_ascend/models/kimi_k3.py',
 'source/alignment_runtime/kda_alignment.py','source/alignment_runtime/kimi_scoring.py',
 'source/alignment_runtime/kda_trace.py','source/alignment_runtime/alignment_audit.py',
 'source/FSDPTurbo/fsdp_turbo/ops/kda.py',
 'source/FSDPTurbo/fsdp_turbo/ops/npu/lightning_indexer.py',
 'source/FSDPTurbo/fsdp_turbo/ops/npu/sparse_flash_attention.py',
 'source/vllm-ascend/vllm_ascend/ops/kimi_kda.py',
 'source/vllm-ascend/vllm_ascend/worker/v2/model_runner.py',
 'source/vllm/vllm/models/kimi_k3/amd/linear.py',
 'source/verl/verl/utils/kimi_logprob.py',
 'source/verl/verl/workers/engine/fsdp/transformer_impl.py',
 'model/modeling_kimi_linear.py','model/moe.py','model/config.json','scripts/launch.sh','scripts/run-alignment.sh',
 'scripts/env-resolved.sh',
]
manifest={'KIMI_ALIGNMENT_ALLTOALL':os.environ.get('KIMI_ALIGNMENT_ALLTOALL','0'),'KIMI_ALIGNMENT_DSA_RAW_CHANNELS':os.environ.get('KIMI_ALIGNMENT_DSA_RAW_CHANNELS','0'),'KIMI_ALIGNMENT_DSA_NORM':os.environ.get('KIMI_ALIGNMENT_DSA_NORM','0'),'KIMI_ALIGNMENT_MOE':os.environ.get('KIMI_ALIGNMENT_MOE','0'),'run':run,'time':datetime.datetime.now().astimezone().isoformat(),
          'KIMI_ALIGNMENT_RECURRENT_KDA':'1','KIMI_ALIGNMENT_V2_SCORING':'1','KIMI_ALIGNMENT_CONV_SPLIT_SILU':os.environ.get('KIMI_ALIGNMENT_CONV_SPLIT_SILU','0'),
          'KIMI_ALIGNMENT_SPLIT_GATED_NORM':os.environ.get('KIMI_ALIGNMENT_SPLIT_GATED_NORM','0'),'KIMI_ALIGNMENT_SPLIT_INPUT_NORM':os.environ.get('KIMI_ALIGNMENT_SPLIT_INPUT_NORM','0'),'KIMI_ALIGNMENT_VISION_FA':os.environ.get('KIMI_ALIGNMENT_VISION_FA','0'),'KIMI_ALIGNMENT_FULL_O_PROJ':os.environ.get('KIMI_ALIGNMENT_FULL_O_PROJ','0'),'KIMI_ALIGNMENT_DENSE_MLP':os.environ.get('KIMI_ALIGNMENT_DENSE_MLP','0'),'KIMI_ALIGNMENT_NATIVE_ATTN_RES':os.environ.get('KIMI_ALIGNMENT_NATIVE_ATTN_RES','0'),'KIMI_PACKAGE_METADATA_CACHE_EPOCH':os.environ.get('KIMI_PACKAGE_METADATA_CACHE_EPOCH'), 'sha256':{},'validation_scope':'4 layers BF16, FSDP8/EP8, rollout TP8/EP8, two steps'}
for relative in files:
    path=w/relative
    manifest['sha256'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    target=root/'source_snapshot'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==path.read_bytes(),'Existing snapshot differs'
    else:shutil.copyfile(path,target)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
p=w/'results/current-status.json';s=json.loads(p.read_text())
s.update(time=manifest['time'],run=run,status='starting_full_two_step_validation',target_met=False,
         sha256=manifest['sha256'],next='Independent old/grad/rollout comparison and first-divergence traces')
s.pop('current_run_exit_code',None)
s.pop('interruption',None)
s['candidate_tests']['exact_v2_scoring_and_entropy']='results/kimi-v2-scoring-validation.json'
p.write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({'run':run,'snapshotted_files':len(files),'target_met':False}))
