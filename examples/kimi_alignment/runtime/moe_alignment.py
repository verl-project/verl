"""Kimi BF16 routing boundaries observed in the actual FSDPTurbo autocast path."""
import torch
import torch.nn.functional as F

def training_gate_logits(hidden,weight):
    assert hidden.dtype==weight.dtype==torch.bfloat16
    return F.linear(hidden,weight).float()

def training_routing(gating_output,bias,topk,renormalize,scaling_factor,
                     scoring_func='sigmoid',num_expert_group=1,topk_group=1):
    logits=gating_output.to(torch.bfloat16)
    assert scoring_func=='sigmoid','Only sigmoid Kimi routing has fixed-input evidence'
    scores=logits.sigmoid()
    choice=scores+bias.to(torch.bfloat16).unsqueeze(0)
    if num_expert_group>1 and num_expert_group>topk_group:
        group_scores=choice.view(choice.shape[0],num_expert_group,-1).topk(2,dim=-1)[0].sum(-1)
        gi=group_scores.topk(topk_group,dim=-1,sorted=False)[1]
        mask=torch.zeros_like(group_scores).scatter_(1,gi,1).unsqueeze(-1)
        mask=mask.expand(-1,num_expert_group,choice.shape[1]//num_expert_group).reshape_as(choice)
        choice=choice.masked_fill(~mask.bool(),float('-inf'))
    ids=choice.topk(topk,dim=-1,sorted=False)[1]
    # NPU autocast promotes gather/normalization to FP32 after BF16 sigmoid.
    weights=scores.float().gather(1,ids)
    if topk>1 and renormalize:weights=weights/(weights.sum(-1,keepdim=True)+1e-20)
    return weights*scaling_factor,ids
