from pathlib import Path
import os,subprocess,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
PY='/root/autodl-tmp/qil-env/bin/python';MODEL='/root/autodl-tmp/models/Qwen3-0.6B'
ENV=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
status={'state':'running','steps':[]}
def save(): (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
def run(name,command):
 status['current']=name;save();print('START',name,flush=True)
 with (OUT/(name+'.log')).open('w') as log:
  p=subprocess.run([PY,*map(str,command)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 status['steps'].append({'name':name,'returncode':p.returncode});save();print('DONE',name,p.returncode,flush=True)
try:
 for name,sha in json.loads((ROOT/'manifest.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 run('backend-probe',[ROOT/'backend_probe.py'])
 for backend,mode in [('math','batched'),('math','serial'),('efficient','batched'),('auto','batched')]:
  for lengths,label in [([17,41,83,128],'short'),([63,127,255,511],'long')]:
   name=f'cached-{backend}-{mode}-{label}'
   run(name,['-m','benchmarks.validate_batched','--model',MODEL,'--workload',ROOT/'workload.json','--dtype','bfloat16','--reference-attention','sdpa','--sdpa-backend',backend,'--reference-cache','--decode-mode',mode,'--prompt-lengths',*lengths,'--output',OUT/(name+'.json')])
 status['state']='diagnostics_completed'
except Exception as e:status['state']='failed';status['error']=repr(e);raise
finally:
 save()
 with zipfile.ZipFile(ROOT.parent/(ROOT.name+'-results.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():
   if p.is_file():z.write(p,p.name)
 print('ARCHIVED',flush=True)
