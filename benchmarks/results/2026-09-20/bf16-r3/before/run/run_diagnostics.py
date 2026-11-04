from pathlib import Path
import os,subprocess,time,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
PY='/root/autodl-tmp/qil-env/bin/python';MODEL='/root/autodl-tmp/models/Qwen3-0.6B'
ENV=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
status={'state':'running','steps':[]}
def save(): (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
def run(name,command,allow_failure=False):
    status['current']=name;save();print('START',name,flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        p=subprocess.run([PY,*map(str,command)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    status['steps'].append({'name':name,'returncode':p.returncode});save();print('DONE',name,p.returncode,flush=True)
    if p.returncode and not allow_failure:raise RuntimeError(name)
try:
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for length in (128,512):
        run('layers'+str(length),['-m','benchmarks.diagnose_bf16','--model',MODEL,'--workload',ROOT/'workload.json','--length',length,'--output',OUT/('layers'+str(length)+'.json')])
    for ref,backend in [('eager','auto'),('sdpa','auto'),('sdpa','math'),('sdpa','efficient')]:
        for lengths,label in [([17,41,83,128],'short'),([63,127,255,511],'long')]:
            name=f'{ref}-{backend}-{label}'
            run(name,['-m','benchmarks.validate_batched','--model',MODEL,'--workload',ROOT/'workload.json','--dtype','bfloat16','--reference-attention',ref,'--sdpa-backend',backend,'--prompt-lengths',*lengths,'--output',OUT/(name+'.json')],allow_failure=True)
    status['state']='diagnostics_completed'
except Exception as e:
    status['state']='failed';status['error']=repr(e);raise
finally:
    save()
    with zipfile.ZipFile(ROOT.parent/(ROOT.name+'-results.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():z.write(p,p.name)
    print('ARCHIVED',flush=True)
