"""Record first observed BF16 greedy divergence without changing tolerances."""
import argparse,json,hashlib
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from qil import Engine,Qwen3


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--workload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=json.loads(args.workload.read_text());source=max((c['prompts'] for c in data['performance'] if c['batch']==4),key=lambda ps:len(ps[0]))
 prompts=[(p*((n+len(p)-1)//len(p)))[:n] for p,n in zip(source,[32,128,512,2048])]
 logits={};sequences={}
 for dtype,modes in [('bfloat16',['dense','grouped','paged']),('float32',['dense'])]:
  model=Qwen3.from_directory(args.model,device='cuda',dtype=getattr(torch,dtype))
  for mode in modes:
   name=dtype+'-'+mode;engine=Engine(model,num_pages=256,page_size=16,token_budget=512,prefill_chunk=128,max_active=4,decode_attention=mode)
   original=model.forward_decode
   def record(tokens,**kwargs):
    out=original(tokens,**kwargs)
    for row,(rid,start) in enumerate(zip(kwargs['request_ids'],kwargs['starts'])):
     if rid==0 and start==38:logits[name]=out[row].float().cpu()
    return out
   model.forward_decode=record
   try:
    with sdpa_kernel(SDPBackend.MATH):
     result=engine.generate(prompts,max_new_tokens=10,ignore_eos=True)
    sequences[name]=[r['token_ids'] for r in result]
    assert not engine.pool.tables
   finally:model.forward_decode=original;engine.close()
  del model,engine
  torch.cuda.empty_cache()
 summary={}
 for name,x in logits.items():
  vals,ids=x.topk(5)
  summary[name]=dict(top_token_ids=ids.tolist(),top_logits=vals.tolist(),margin=float(vals[0]-vals[1]),
                     candidate_logits={str(i):float(x[i]) for i in (6145,15)})
 a,b=logits['bfloat16-dense'],logits['bfloat16-paged']
 report=dict(first_observed_divergence=dict(request_id=0,prompt_length=32,generated_token_index=7,decode_position=38),
             summary=summary,sequences=sequences,same_prefix_before_divergence=sequences['bfloat16-dense'][0][:7]==sequences['bfloat16-paged'][0][:7],
             max_abs_error=float((a-b).abs().max()),within_logit_tolerance=bool(torch.allclose(a,b,atol=.25,rtol=.02)),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='sequences'},indent=2))
if __name__=='__main__':main()
