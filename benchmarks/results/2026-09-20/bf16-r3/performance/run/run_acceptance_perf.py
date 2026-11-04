from pathlib import Path
import os,subprocess,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
PY='/root/autodl-tmp/qil-env/bin/python';VLLM='/root/autodl-tmp/vllm-env/bin/python';MODEL='/root/autodl-tmp/models/Qwen3-0.6B'
ENV=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
status={'state':'running','steps':[]}
def save():(OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
def run(name,python,command,diagnostic=False):
 status['current']=name;save();print('START',name,flush=True)
 with (OUT/(name+'.log')).open('w') as log:
  p=subprocess.run([python,*map(str,command)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 status['steps'].append({'name':name,'returncode':p.returncode});save();print('DONE',name,p.returncode,flush=True)
 if p.returncode and not diagnostic:raise RuntimeError(name+' failed')
try:
 for name,sha in json.loads((ROOT/'manifest.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 gates=[]
 for backend in ('math','efficient'):
  for length in ('short','long'):
   path=BASE/'qil-bf16-20260920-r3-cached/results'/f'cached-{backend}-batched-{length}.json'
   r=json.loads(path.read_text());assert r['passed'],str(path)
   for name in ('qil/model.py','qil/cache.py','qil/engine.py','qil/config.py','qil/scheduler.py'):
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==r['source_sha256'][name],name
   gates.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'passed':True})
 assert json.loads((BASE/'qil-bf16-20260920-r3-fixed/results/fp32-regression.json').read_text())['passed']
 (OUT/'acceptance_gate.json').write_text(json.dumps({'scope':'Fixed same SDPA backend and cached HF reference; no tolerance changes','gates':gates},indent=2)+'\n')
 for lengths,label in [([17,41,83,128],'short'),([63,127,255,511],'long')]:
  run('full-reduction-'+label,PY,['-m','benchmarks.validate_batched','--model',MODEL,'--workload',ROOT/'workload.json','--dtype','bfloat16','--reference-attention','sdpa','--sdpa-backend','math','--full-bf16-reduction','--prompt-lengths',*lengths,'--output',OUT/('full-reduction-'+label+'.json')],diagnostic=True)
 for name,python,backend,dtype,flags in [('qil-bf16',PY,'qil','bfloat16',['--batch-decode','--selective-logits','--sdpa-backend','efficient']),('qil-fp32',PY,'qil','float32',['--batch-decode','--selective-logits']),('vllm-bf16',VLLM,'vllm','bfloat16',[])]:
  run(name,python,['-m','benchmarks.compare_independent','--backend',backend,'--model',MODEL,'--workload',ROOT/'workload.json','--dtype',dtype,'--runs','5','--output',OUT/(name+'.json'),*flags])
 status['state']='completed'
except Exception as e:status['state']='failed';status['error']=repr(e);raise
finally:
 save()
 with zipfile.ZipFile(BASE/'qil-bf16-20260920-r3-all-results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name,folder in [('before','qil-bf16-20260920-r3'),('fixed','qil-bf16-20260920-r3-fixed'),('cached','qil-bf16-20260920-r3-cached'),('performance',ROOT.name)]:
   d=BASE/folder
   for p in (d/'results').iterdir():
    if p.is_file() and p.suffix in ('.json','.log'):z.write(p,name+'/'+p.name)
   for p in d.iterdir():
    if p.is_file() and (p.suffix in ('.py','.log') or p.name in ('manifest.json','workload.json')):z.write(p,name+'/run/'+p.name)
 print('ALL_RESULTS_ARCHIVED',flush=True)
