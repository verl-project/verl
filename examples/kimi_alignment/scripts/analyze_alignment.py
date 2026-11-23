"""Read-only analysis of task-owned raw alignment evidence."""
import sys,json,hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
w=Path('/mnt/share/y00938198/kimi-k31/test_wxx_skill')
run=sys.argv[1];root=w/'results'/run/'audit'
def read(p):return torch.load(p,map_location='cpu',weights_only=True)
def values(x):return x['values'] if isinstance(x,dict) and 'values' in x else x
def stats(a,b):
 a=a.float();b=b.float()
 if a.shape!=b.shape:return {'shape_mismatch':[list(a.shape),list(b.shape)],'passed':False}
 f=torch.isfinite(a)&torch.isfinite(b);d=(a-b).abs();n=a.numel()
 return {'numel':n,'nonfinite':int((~f).sum()),'different':int((a!=b).sum()),'mean_abs':float(d.mean()) if n else None,'max_abs':float(d.max()) if n else None,'passed':bool(n and f.all() and (a==b).all())}
report={'run':run,'global_audits':[json.loads(p.read_text()) for p in sorted(root.glob('*.json'))],'prefix_probes':[],'actor_vs_reference':[],'actor_old_vs_grad':[]}
for p in sorted((root/'prefix_probe').glob('*.pt')):
 x=read(p);r={'file':p.name,'rank':x['rank'],'prefix_length':x['prefix_length'],'expanded_prefix_length':x.get('expanded_prefix_length'),'repeat':stats(x['original_full_logits'],x['repeat_full_logits']),'prefix':stats(x['original_full_logits'],x['prefix_logits']),'boundaries':{}}
 target=x['target_token'];r['target_token']=target
 r['diagnostic_cpu_logprobs']={k:float(x[k].float().log_softmax(-1)[target]) for k in ('original_full_logits','repeat_full_logits','prefix_logits')}
 traces=x['boundary_traces']
 for name,a in traces.get('repeat_full',{}).items():
  b=traces.get('prefix_only',{}).get(name)
  if b is not None:r['boundaries'][name]=stats(a,b)
 report['prefix_probes'].append(r)
groups={}
for p in sorted((root/'workers').glob('*.pt')):
 x=read(p);ids=x['input_ids'];key=hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
 step=x.get('batch_global_steps');key+=str(step)
 role='ref' if x['engine_forward_only'] else ('grad' if x['grad_enabled'] else 'actor')
 groups.setdefault(key,{})[role]=(p,x)
for key,group in groups.items():
 if 'actor' not in group:continue
 pa,a=group['actor'];av=values(a['log_probs'])
 for role,out in [('ref','actor_vs_reference'),('grad','actor_old_vs_grad')]:
  if role not in group:continue
  pb,b=group[role];bv=values(b['log_probs']);response=values(a['batch_responses']);n=response.numel()
  r={'actor_file':pa.name,'other_file':pb.name,'global_steps':a.get('batch_global_steps'),'full_sequence':stats(av,bv),'response':stats(av[-n-1:-1],bv[-n-1:-1]),'other_grad_enabled':b['grad_enabled'],'other_requires_grad':b['logprobs_require_grad']}
  report[out].append(r)
report['vllm_step_files']=[p.name for p in sorted((root/'vllm_steps').glob('*.pt'))]
path=w/'results'/run/'analysis.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(path),'global_audits':report['global_audits'],'prefix_probes':report['prefix_probes'],'actor_ref_pairs':len(report['actor_vs_reference']),'actor_grad_pairs':len(report['actor_old_vs_grad']),'actor_ref_max':max((r['response']['max_abs'] for r in report['actor_vs_reference']),default=None),'actor_grad_max':max((r['response']['max_abs'] for r in report['actor_old_vs_grad']),default=None),'vllm_step_files':report['vllm_step_files']},indent=2))
