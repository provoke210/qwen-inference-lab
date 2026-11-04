"""Layerwise BF16 diagnostic. No tolerances are changed and no speed is measured."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as hf
from qil import Qwen3, Engine
from qil.model import DecoderBlock


def difference(a, b):
    a,b=a.float(),b.float()
    d=(a-b).abs()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                exact=torch.equal(a,b), allclose=torch.allclose(a,b,atol=.25,rtol=.02),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def capture(model, tokens, native, kernel):
    values, inputs, handles = {}, {}, []
    current = [0]
    def add(name, module):
        def hook(m, args, result):
            value=result[0] if isinstance(result, tuple) else result
            values[name]=value.detach().cpu().reshape(len(tokens),-1)
            inputs[name]=args[0].detach().cpu()
        handles.append(module.register_forward_hook(hook))
    blocks=model.blocks if native else model.model.layers
    for i,b in enumerate(blocks):
        def before(m,args,i=i):current[0]=i
        handles.append(b.register_forward_pre_hook(before))
        mapping = dict(norm1=b.before_attention,q=b.query,k=b.key,v=b.value,qn=b.query_norm,
                       kn=b.key_norm,attn_out=b.attention_output,norm2=b.before_mlp,
                       gate=b.gate,up=b.expand,mlp=b.contract,block=b) if native else dict(
                       norm1=b.input_layernorm,q=b.self_attn.q_proj,k=b.self_attn.k_proj,
                       v=b.self_attn.v_proj,qn=b.self_attn.q_norm,kn=b.self_attn.k_norm,
                       attn_out=b.self_attn.o_proj,norm2=b.post_attention_layernorm,
                       gate=b.mlp.gate_proj,up=b.mlp.up_proj,mlp=b.mlp.down_proj,block=b)
        for name,module in mapping.items():add(f'{i}.{name}',module)
    add('final_norm',model.final_norm if native else model.model.norm)
    add('head',model.head if native else model.lm_head)
    rotate=DecoderBlock.rotate
    hf_rotate=hf.apply_rotary_pos_emb
    def native_rope(x, angles):
        result=rotate(x,angles)
        name='rq' if f'{current[0]}.rq' not in values else 'rk'
        values[f'{current[0]}.{name}']=result.cpu().reshape(len(tokens),-1)
        if current[0]==0:
            values['cos']=angles.cos().to(x.dtype).cpu()
            values['sin']=angles.sin().to(x.dtype).cpu()
        return result
    def reference_rope(q,k,cos,sin,*args,**kwargs):
        qr,kr=hf_rotate(q,k,cos,sin,*args,**kwargs)
        values[f'{current[0]}.rq']=qr[0].transpose(0,1).cpu().reshape(len(tokens),-1)
        values[f'{current[0]}.rk']=kr[0].transpose(0,1).cpu().reshape(len(tokens),-1)
        values['cos']=cos[0,:,:cos.shape[-1]//2].cpu()
        values['sin']=sin[0,:,:sin.shape[-1]//2].cpu()
        return qr,kr
    ctx = sdpa_kernel(SDPBackend.MATH) if kernel=='math' else nullcontext()
    try:
        with torch.inference_mode(),ctx,patch.object(DecoderBlock,'rotate',staticmethod(native_rope)),patch.object(hf,'apply_rotary_pos_emb',reference_rope):
            if native:
                owner=Engine(model,num_pages=256,page_size=16)
                owner.pool.admit(0,len(tokens));owner.pool.ensure(0,len(tokens))
                model(tokens,start=0,pool=owner.pool,request_id=0)
                owner.pool.release(0)
            else:
                model(torch.tensor([tokens],device=next(model.parameters()).device),use_cache=False)
    finally:
        for h in handles:h.remove()
    return values,inputs


def compare(left,right):
    return {name:difference(left[name],right[name]) for name in left.keys() & right.keys()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--workload',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--length',type=int,default=128)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    workload=json.loads(args.workload.read_text())
    seed=next(c['prompts'][0] for c in workload['performance'] if c['batch']==4)
    tokens=(seed*((args.length+len(seed)-1)//len(seed)))[:args.length]
    all_values,all_inputs={},{}
    report=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,
                dtype='bfloat16',length=len(tokens),token_ids=tokens,comparisons={},isolated={},
                source_sha256={'qil/model.py':hashlib.sha256(Path('qil/model.py').read_bytes()).hexdigest(),
                               'diagnose_bf16.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                reference_sources={name:inspect.getsource(obj) for name,obj in [
                    ('RMSNorm',hf.Qwen3RMSNorm),('RoPE',hf.apply_rotary_pos_emb),
                    ('RotaryEmbedding',hf.Qwen3RotaryEmbedding),('attention',hf.Qwen3Attention.forward)]})
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    for name,attention,kernel in [('hf_eager','eager','auto'),('hf_sdpa','sdpa','auto'),('hf_math','sdpa','math')]:
        print('START',name,flush=True)
        ref=Qwen3ForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation=attention,
                                           local_files_only=True).to(args.device).eval()
        all_values[name],all_inputs[name]=capture(ref,tokens,False,kernel)
        del ref;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    model=Qwen3.from_directory(args.model,dtype=torch.bfloat16,device=args.device)
    for name,kernel in [('qil_sdpa','auto'),('qil_math','math')]:
        print('START',name,flush=True)
        all_values[name],all_inputs[name]=capture(model,tokens,True,kernel)
    for a,b in [('qil_sdpa','hf_eager'),('qil_sdpa','hf_sdpa'),('qil_math','hf_math'),('hf_sdpa','hf_eager'),('hf_math','hf_eager')]:
        comparisons=compare(all_values[a],all_values[b])
        report['comparisons'][a+'__'+b]=comparisons
        print(a,b,'head',comparisons['head'],flush=True)
    # Replay each native component with exactly the reference component's input.
    # This separates a local operation difference from differences propagated from earlier layers.
    with torch.inference_mode():
        for i,b in enumerate(model.blocks):
            for name,module in dict(norm1=b.before_attention,q=b.query,k=b.key,v=b.value,qn=b.query_norm,
                                    kn=b.key_norm,attn_out=b.attention_output,norm2=b.before_mlp,
                                    gate=b.gate,up=b.expand,mlp=b.contract).items():
                key=f'{i}.{name}'
                x=all_inputs['hf_math'][key].to(args.device)
                actual=module(x).cpu().reshape(len(tokens),-1)
                report['isolated'][key]=difference(actual,all_values['hf_math'][key])
        for i,b in enumerate(model.blocks):
            x=all_values['hf_math'][f'{i}.norm2'].to(args.device)
            actual=b.contract(F.silu(b.gate(x))*b.expand(x)).cpu()
            report['isolated'][f'{i}.mlp_full']=difference(actual,all_values['hf_math'][f'{i}.mlp'])
            cfg=model.config
            def layout(key,heads):
                return all_values['hf_math'][f'{i}.{key}'].to(args.device).reshape(len(tokens),heads,cfg.head_dim).transpose(0,1)[None]
            q,k,v=layout('rq',cfg.num_attention_heads),layout('rk',cfg.num_key_value_heads),layout('v',cfg.num_key_value_heads)
            rep=cfg.num_attention_heads//cfg.num_key_value_heads
            k,v=k.repeat_interleave(rep,dim=1),v.repeat_interleave(rep,dim=1)
            mask=torch.ones(len(tokens),len(tokens),dtype=torch.bool,device=args.device).tril()
            with sdpa_kernel(SDPBackend.MATH):
                attn=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=cfg.head_dim**-.5)
            attn=attn[0].transpose(0,1).reshape(len(tokens),-1).cpu()
            report['isolated'][f'{i}.attention']=difference(attn,all_inputs['hf_math'][f'{i}.attn_out'].reshape(len(tokens),-1))
        angles=model.rotary_angles(torch.arange(len(tokens),device=args.device))
        for i in range(len(model.blocks)):
            for name,heads,rotated in [('qn',model.config.num_attention_heads,'rq'),('kn',model.config.num_key_value_heads,'rk')]:
                x=all_values['hf_math'][f'{i}.{name}'].to(args.device).reshape(len(tokens),heads,model.config.head_dim)
                actual=DecoderBlock.rotate(x,angles).cpu().reshape(len(tokens),-1)
                report['isolated'][f'{i}.{rotated}']=difference(actual,all_values['hf_math'][f'{i}.{rotated}'])
    report['complete']=True;save()
    print('COMPLETE isolated nonexact',[k for k,v in report['isolated'].items() if not v['exact']],flush=True)

if __name__=='__main__':main()
