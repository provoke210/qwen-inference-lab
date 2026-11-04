"""Isolate KV attention cost and paired full-engine latency; preserve every sample."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from qil import Engine, Qwen3
from qil.cache import PagePool
from qil.attention import decode_attention


def save(path, report):
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


def micro(dtype):
    rows=[]
    for lengths in ([128]*4,[512]*4,[17,63,255,511],[32,128,512,2048]):
        outputs=[]
        for mode in ('dense','grouped','paged'):
            torch.manual_seed(51)
            pool=PagePool(layers=1,pages=256,page_size=16,kv_heads=8,head_dim=128,
                          dtype=dtype,device='cuda',decode_attention=mode)
            pool.keys.normal_(); pool.values.normal_()
            for rid,n in enumerate(lengths):
                assert pool.admit(rid,n)
                pool.ensure(rid,n)
                pool.written[rid][0]=n-1
            batch=pool.prepare_decode(range(4),[n-1 for n in lengths])
            for rid,n in enumerate(lengths):pool.written[rid][0]=n
            q=torch.randn(4,16,128,device='cuda',dtype=dtype)
            run=lambda:decode_attention(q,pool,batch,0)
            for _ in range(5):out=run()
            outputs.append(out.clone())
            torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            out=run(); torch.cuda.synchronize()
            temporary=max(0,torch.cuda.max_memory_allocated()-baseline)
            samples=[]
            for _ in range(7):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):run()
                end.record();end.synchronize()
                samples.append(start.elapsed_time(end)*1000/30)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                run();torch.cuda.synchronize()
            names=[event.key for event in prof.key_averages()]
            slots=(batch.read_slots.numel() if mode=='dense' else
                   sum(b.read_slots.numel() for _,b in batch.groups) if mode=='grouped' else 0)
            rows.append(dict(lengths=list(lengths),mode=mode,median_us=statistics.median(samples),
                             samples_us=samples,temporary_bytes=temporary,materialized_kv_slots=slots,
                             profiler_operations=names))
            del pool,batch,q,out
        for other in outputs[1:]:
            torch.testing.assert_close(outputs[0],other,atol=.008 if dtype==torch.bfloat16 else 2e-5,
                                       rtol=.008 if dtype==torch.bfloat16 else 2e-4)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--dtype',choices=['float32','bfloat16'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sdpa-backend',choices=['math','efficient'],default='efficient')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype=getattr(torch,args.dtype);workload=json.loads(args.workload.read_text())
    root=Path(__file__).resolve().parents[1]
    sources=[p for p in (root/'qil').glob('*.py') if p.name!='Untitled-1.py']+[Path(__file__)]
    report=dict(dtype=args.dtype,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                sdpa_backend=args.sdpa_backend,warmups=1,runs=5,complete=False,
                timing='Full engine prefill+decode; median paired wall time; no load/tokenization/JIT',
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                micro=[],performance=[])
    with torch.inference_mode(),sdpa_kernel({'math':SDPBackend.MATH,'efficient':SDPBackend.EFFICIENT_ATTENTION}[args.sdpa_backend]):
        report['micro']=micro(dtype);save(args.output,report)
        model=Qwen3.from_directory(args.model,device='cuda',dtype=dtype)
        engines={mode:Engine(model,num_pages=256,page_size=16,token_budget=512,prefill_chunk=128,
                             max_active=4,batch_decode=True,selective_logits=True,decode_attention=mode)
                 for mode in ('dense','grouped','paged')}
        source=max((c['prompts'] for c in workload['performance'] if c['batch']==4),key=lambda ps:len(ps[0]))
        cases=[c for c in workload['performance'] if c['batch']==4]
        for lengths in ([17,63,255,511],[32,128,512,2048]):
            cases.append(dict(name='mixed_'+'_'.join(map(str,lengths)),max_new_tokens=32,
                              prompts=[(ps*((n+len(ps)-1)//len(ps)))[:n] for ps,n in zip(source,lengths)]))
        # Also exercise the new single-request route.
        cases += [c for c in workload['performance'] if c['batch']==1]
        for case in cases:
            samples={mode:[] for mode in engines}; outputs={mode:[] for mode in engines}
            def run(mode):
                e=engines[mode]
                result=e.generate(case['prompts'],max_new_tokens=32,ignore_eos=True)
                assert e.idle and not e.pool.tables and len(e.pool.free)==256
                assert all(r['state']=='completed' for r in result)
                return [r['token_ids'] for r in result]
            for mode in engines:run(mode)
            for iteration in range(5):
                order=list(engines);order=order[iteration%3:]+order[:iteration%3]
                for mode in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    output=run(mode);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    samples[mode].append(elapsed);outputs[mode].append(output)
            match=all(out==outputs['dense'][0] for seq in outputs.values() for out in seq)
            row=dict(name=case['name'],lengths=list(map(len,case['prompts'])),new_tokens=32,
                     samples_s=samples,median_s={m:statistics.median(v) for m,v in samples.items()},
                     output_token_ids_by_run=outputs,all_outputs_match=match)
            row['output_tokens_per_s']={m:len(case['prompts'])*32/v for m,v in row['median_s'].items()}
            report['performance'].append(row);save(args.output,report)
            print(json.dumps({k:v for k,v in row.items() if k not in ('output_token_ids_by_run','samples_s')}),flush=True)
        for engine in engines.values():engine.close()
    report['complete']=True
    report['generation_parity_passed']=all(row['all_outputs_match'] for row in report['performance'])
    save(args.output,report)
    if not report['generation_parity_passed']:
        raise SystemExit('Generation parity failed; retain timings as diagnostic data, not accepted speedups')


if __name__=='__main__':main()
