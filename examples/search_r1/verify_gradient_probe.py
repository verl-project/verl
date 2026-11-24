"""Four-rank FSDP smoke of the diagnostic lifecycle; run with torchrun."""
import json
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .hidden_gradient_diagnostics import measure_gradients


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl')
    torch.manual_seed(42)
    model=FSDP(torch.nn.Linear(128,8,bias=False).cuda(),device_id=rank)
    opt=torch.optim.SGD(model.parameters(),lr=.1)
    before=[p.detach().clone() for p in model.parameters()]
    class Engine:
        module=model
        scaler=None
        def optimizer_zero_grad(self):opt.zero_grad(set_to_none=True)
        def get_data_parallel_group(self):return torch.distributed.group.WORLD
        def forward_backward_batch(self,data,loss):
            for _ in range(2):
                scalar=model(torch.ones(1,128,device='cuda')).sum()*(rank+1)
                if loss.args[0]=='hidden':scalar=scalar*-.001
                scalar.backward()
    engine=Engine()
    result=measure_gradients(engine,None,partial(lambda:None,config=None))
    assert abs(result['turn_hidden/gradient/hidden_to_grpo_ratio']-.001)<1e-7,result
    assert result['turn_hidden/gradient/cosine']<-.99999,result
    assert abs(result['turn_hidden/gradient/grpo_norm']-160)<.01,result
    for a,b in zip(before,model.parameters()):
        torch.testing.assert_close(a,b)
        assert b.grad is None
    if rank==0:
        Path(os.environ['RUN_DIR'],'gradient-probe-preflight.json').write_text(json.dumps({'passed':True,'metrics':result},indent=2))
    torch.distributed.destroy_process_group()

if __name__=='__main__':main()
