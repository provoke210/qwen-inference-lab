import json,torch
from pathlib import Path
from transformers import Qwen3ForCausalLM
from qil import Engine,Qwen3
ROOT=Path(__file__).resolve().parent
model_path='/root/autodl-tmp/models/Qwen3-0.6B'
tokens=json.loads((ROOT/'workload.json').read_text())['performance'][0]['prompts'][0]
results={}
for label in ('qil','hf'):
    m=Qwen3.from_directory(model_path,device='cuda',dtype=torch.bfloat16) if label=='qil' else Qwen3ForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        if label=='qil':
            e=Engine(m,num_pages=16,page_size=16);e.pool.admit(0,len(tokens));e.pool.ensure(0,len(tokens));m(tokens,start=0,pool=e.pool,request_id=0);e.pool.release(0)
        else:m(torch.tensor([tokens],device='cuda'),use_cache=False)
        torch.cuda.synchronize()
    results[label]=[dict(name=k.key,count=k.count) for k in profile.key_averages() if 'attention' in k.key]
    if label=='qil':del e
    del m
    import gc
    gc.collect();torch.cuda.empty_cache()
(ROOT/'results/backend-probe.json').write_text(json.dumps(results,indent=2)+'\n')
print(results,flush=True)
