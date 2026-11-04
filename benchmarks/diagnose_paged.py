"""Replay identical real-model attention inputs to distinguish BF16 backends."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from qil import Engine,Qwen3
from qil.paged_attention import paged_attention
import qil.model as runtime


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--workload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=json.loads(a.workload.read_text());source=max((c['prompts'] for c in data['performance'] if c['batch']==4),key=lambda ps:len(ps[0]))
 prompts=[p[:n] for p,n in zip(source,[17,63,255,511])]
 model=Qwen3.from_directory(a.model,device='cuda',dtype=torch.bfloat16)
 original=runtime.decode_attention;rows=[]
 def replay(q,pool,batch,layer):
  expected=original(q,pool,batch,layer)
  actual=paged_attention(q,pool.keys[layer],pool.values[layer],batch.page_table,batch.lengths,pool.page_size,max(batch.starts)+1)
  k,v=pool.read_decode(batch,layer);r=q.shape[1]//k.shape[2]
  k=k.repeat_interleave(r,dim=2).transpose(1,2);v=v.repeat_interleave(r,dim=2).transpose(1,2)
  with sdpa_kernel(SDPBackend.MATH):
   math=F.scaled_dot_product_attention(q.unsqueeze(2),k,v,attn_mask=batch.valid_tokens[:,None,None,:],scale=q.shape[-1]**-.5).squeeze(2)
  def compare(x,y):
   d=(x.float()-y.float()).abs()
   return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),different=int((x!=y).sum()),elements=x.numel())
  rows.append(dict(layer=layer,lengths=[s+1 for s in batch.starts],paged_vs_efficient=compare(actual,expected),paged_vs_math=compare(actual,math),math_vs_efficient=compare(math,expected)))
  return expected
 runtime.decode_attention=replay
 try:
  with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
   e=Engine(model,num_pages=128,page_size=16,token_budget=512,prefill_chunk=128,max_active=4)
   result=e.generate(prompts,max_new_tokens=4,ignore_eos=True)
   assert not e.pool.tables
 finally:runtime.decode_attention=original
 a.output.write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
 for key in ('paged_vs_efficient','paged_vs_math','math_vs_efficient'):
  print(key,'max',max(r[key]['max_abs'] for r in rows),'different',sum(r[key]['different'] for r in rows),'elements',sum(r[key]['elements'] for r in rows))
if __name__=='__main__':main()
