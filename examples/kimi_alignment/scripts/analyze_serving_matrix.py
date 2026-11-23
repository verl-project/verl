"""Verify fixed-input same-side A/A, BI and full-prefill/decode controls."""
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(2)
root=Path(sys.argv[1])
folder=root/"audit/serving_invariance"
assert json.loads((folder/"matrix-complete.json").read_text())["completed"]
outputs={x["id"]:x for x in json.loads((folder/"matrix-output.json").read_text())}
records={}
for path in folder.glob("alignment-probe-*.pt"):
    rec=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    key=rec["request"]
    # AsyncLLM can append internal request suffixes. Prefix labels are unique.
    label=next((x for x in outputs if key.startswith(x)),None)
    if label is not None:
        pos=int(rec["positions"][-1])
        assert pos not in records.setdefault(label,{})
        records[label][pos]=rec

def difference(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    finite=a.isfinite()&b.isfinite()
    bits={torch.float32:torch.int32,torch.bfloat16:torch.int16,torch.float16:torch.int16}
    return {"different":int((a!=b).sum()),
            "different_bits":int((a.contiguous().view(bits[a.dtype])!=b.contiguous().view(bits[b.dtype])).sum()),"finite_mask_different":int((a.isfinite()!=b.isfinite()).sum()),
            "nan_count":int((a.isnan()|b.isnan()).sum()),
            "maximum":float((a[finite].float()-b[finite].float()).abs().max())}

def history(group,pos):
    history={}
    for p,record in records[group].items():
        if p>pos:continue
        for idx,tok in zip(record["positions"].tolist(),record["ids"].tolist()):
            assert idx not in history
            history[idx]=tok
    assert sorted(history)==list(range(pos+1))
    return [history[i] for i in range(pos+1)]

baseline="alignment-probe-single-target"
base=records[baseline]
result={"scope":"One image/prompt; greedy generated positions; fixed policy, BF16 TP8/EP8.",
        "version":json.loads((folder/"matrix-complete.json").read_text())["version"],
        "comparisons":[],"chunked_prefill":"pending"}
groups=[("repeat","AA"),("mixed","BI"),("reordered","BI")]
if "alignment-probe-batch16-target" in outputs:
    groups.append(("batch16","BI"))
for label,kind in groups:
    group=f"alignment-probe-{label}-target"
    assert len(base)==len(records[group])==len(outputs[baseline]["token_ids"])
    assert outputs[baseline]["weight_version"]==outputs[group]["weight_version"]
    entry={"comparison":kind,"group":label,"positions":[]}
    for pos in sorted(base):
        a,b=base[pos],records[group][pos]
        assert history(baseline,pos)==history(group,pos)
        assert a["positions"].tolist()==b["positions"].tolist()
        entry["positions"].append({"position":pos,"phase":"decode" if a["positions"].numel()==1 else "prefill",
                                  "batches":[len(a["batch_ids"]),len(b["batch_ids"])],
                                  "batch_members":[a["batch_ids"],b["batch_ids"]],
                                  "logits":difference(a["logits"],b["logits"])})
    if kind=="BI":
        assert all(len(x["batch_members"][1])==(16 if label=="batch16" else 2) for x in entry["positions"]), "Intended mixed batch did not occur"
    a,b=outputs[baseline],outputs[group]
    entry["generated_tokens_equal"]=a["token_ids"]==b["token_ids"]
    entry["selected_logprobs"]=difference(torch.tensor(a["logprobs"],dtype=torch.float32),
                                           torch.tensor(b["logprobs"],dtype=torch.float32))
    result["comparisons"].append(entry)
group="alignment-probe-full-prefix-target"
pos,=records[group]
assert history(baseline,pos)==history(group,pos)
result["full_prefill_vs_decode"]={"position":pos,
    "decode_queries":base[pos]["positions"].numel(),"prefill_queries":records[group][pos]["positions"].numel(),
    "logits":difference(base[pos]["logits"],records[group][pos]["logits"])}
assert result["full_prefill_vs_decode"]["decode_queries"]==1
assert result["full_prefill_vs_decode"]["prefill_queries"]>1
result["passed"]=all(
    row["logits"]["different"]==0 and row["logits"]["finite_mask_different"]==0 and row["logits"]["nan_count"]==0
    for entry in result["comparisons"] for row in entry["positions"]
) and all(entry["generated_tokens_equal"] and entry["selected_logprobs"]["different"]==0
          for entry in result["comparisons"]) and result["full_prefill_vs_decode"]["logits"]["different"]==0
if (folder/"scheduler-chunks.jsonl").exists():
    long_full=next(iter(records["alignment-probe-long-full-target"].values()))
    long_chunked=next(iter(records["alignment-probe-long-chunked-target"].values()))
    assert int(long_full["positions"][-1])==int(long_chunked["positions"][-1])
    full_tokens=long_full["ids"].tolist()
    chunks=[]
    for line in (folder/"scheduler-chunks.jsonl").read_text().splitlines():
        for entry in json.loads(line)["requests"]:
            if entry["id"].startswith("alignment-probe-long-chunked-target"):
                chunks.append(entry)
    # Verify the exact scheduled intervals form the full causal prefill.
    computed=0
    scheduled_tokens=[]
    for entry in chunks:
        assert entry["computed"]==computed
        assert len(entry["tokens"])==entry["scheduled"]
        scheduled_tokens.extend(entry["tokens"])
        computed+=entry["scheduled"]
    assert len(chunks)>1 and computed==len(full_tokens)
    assert scheduled_tokens==full_tokens
    result["chunked_prefill"]={"chunks":chunks,
        "prefill_tokens":len(full_tokens),
        "logits":difference(long_full["logits"],long_chunked["logits"])}
    result["passed"] = result["passed"] and result["chunked_prefill"]["logits"]["different"]==0
mixed=result["comparisons"][1]["positions"][0]["batch_members"][1]
reordered=result["comparisons"][2]["positions"][0]["batch_members"][1]
result["actual_target_row"] = {
    "mixed":next(i for i,x in enumerate(mixed) if "-target-" in x),
    "reordered":next(i for i,x in enumerate(reordered) if "-target-" in x)}
result["request_order_changed"] = len(set(result["actual_target_row"].values()))==2
result["numerical_checks_passed"]=result.pop("passed")
result["coverage"]={
    "generated_positions":len(base),
    "batch16":any(c["group"]=="batch16" for c in result["comparisons"]),
    "request_order_changed":result["request_order_changed"],
    "chunked_prefill":isinstance(result["chunked_prefill"],dict),
}
result["complete_matrix_passed"]=bool(result["numerical_checks_passed"] and
    result["coverage"]["generated_positions"]==64 and
    result["coverage"]["batch16"] and result["coverage"]["request_order_changed"] and
    result["coverage"]["chunked_prefill"])
(folder/"matrix-analysis.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
