"""Verify the inference export against step-200 rank-zero tensor samples."""

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def tensor_sample(directory, key, shape):
    directory = Path(directory)
    index = directory/'model.safetensors.index.json'
    filename = json.loads(index.read_text())['weight_map'][key] if index.exists() else 'model.safetensors'
    with safe_open(directory/filename, framework='pt', device='cpu') as stream:
        value = stream.get_slice(key)
        slices = tuple(slice(0, min(size, 4) if i == 0 else size) for i,size in enumerate(shape))
        return value[slices].flatten()[:4096]


def main(checkpoint, original, merged, output):
    checkpoint, original, merged = map(Path, [checkpoint,original,merged])
    source = torch.load(checkpoint/'model_world_size_8_rank_0.pt', map_location='cpu', weights_only=False)
    checks = {}
    changed = False
    for key in ['model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight','model.layers.0.input_layernorm.weight']:
        value = source[key]
        local = value._local_tensor if hasattr(value,'_local_tensor') else value
        expected = local.bfloat16()
        # The checkpoint has one-dimensional Shard(0) placements.
        if hasattr(value,'placements'):
            assert len(value.placements)==1 and value.placements[0].is_shard(0),value.placements
        expected = expected[:4].flatten()[:4096]
        exported = tensor_sample(merged,key,local.shape)
        reference = tensor_sample(original,key,local.shape)
        assert torch.equal(expected,exported),key
        differs = not torch.equal(exported,reference)
        changed = changed or differs
        checks[key] = {'checkpoint_sample_equal':True,'differs_from_original_sample':differs}
    assert changed,'Export unexpectedly matches original weights in all sampled tensors'
    a = AutoTokenizer.from_pretrained(original,local_files_only=True)
    b = AutoTokenizer.from_pretrained(merged,local_files_only=True)
    assert a.get_vocab()==b.get_vocab() and a.chat_template==b.chat_template
    assert a.eos_token_id==b.eos_token_id and a.pad_token_id==b.pad_token_id
    assert AutoConfig.from_pretrained(merged,local_files_only=True).model_type=='qwen2'
    shards=[]
    for path in sorted(checkpoint.glob('model_world_size_8_rank_*.pt')):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(16*1024*1024),b''):digest.update(block)
        shards.append({'file':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
    assert len(shards)==8
    report={'verified':True,'checkpoint':str(checkpoint),'checkpoint_step':200,'merged_model':str(merged),
            'source_shards':shards,'tensor_checks':checks,'tokenizer_vocab_and_template_equal':True}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print('MERGED_CHECKPOINT_VERIFIED',json.dumps(checks),flush=True)


if __name__=='__main__':
    main(*sys.argv[1:])
