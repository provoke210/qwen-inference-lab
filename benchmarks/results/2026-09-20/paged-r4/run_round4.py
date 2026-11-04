import json,subprocess,sys,traceback,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=True)
python='/root/autodl-tmp/qil-env/bin/python'
model='/root/autodl-tmp/models/Qwen3-0.6B'
workload=str(root/'workload.json')
status={'state':'running','steps':[]}
def save(): (out/'status.json').write_text(json.dumps(status,indent=2))
def run(name,args):
 status['current']=name;save()
 with (out/(name+'.log')).open('w') as log:
  r=subprocess.run([python,*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 status['steps'].append({'name':name,'returncode':r.returncode});save()
 print(name,r.returncode,flush=True)
 return r.returncode==0
try:
 kernel=run('kernel-tests',['-m','pytest','tests/qil/test_paged_attention.py','-q','--disable-warnings','--junitxml='+str(out/'kernel-tests.xml')])
 if not kernel: raise RuntimeError('Kernel correctness gate failed')
 accepted={}
 for dtype in ('float32','bfloat16'):
  accepted[dtype]=True
  for mode in ('grouped','paged'):
   for group,lengths in [('short',[17,41,83,128]),('long',[63,127,255,511])]:
    name=f'{dtype}-{mode}-{group}'
    ok=run(name,['-m','benchmarks.validate_batched','--model',model,'--workload',workload,
         '--dtype',dtype,'--reference-attention','sdpa','--sdpa-backend','efficient','--reference-cache',
         '--decode-attention',mode,'--prompt-lengths',*map(str,lengths),'--output',str(out/(name+'.json'))])
    accepted[dtype]=accepted[dtype] and ok
  if accepted[dtype]:
   run(dtype+'-profile',['-m','benchmarks.profile_paged','--model',model,'--workload',workload,
                         '--dtype',dtype,'--output',str(out/(dtype+'-profile.json'))])
 status['accepted']=accepted;status['state']='completed';save()
except Exception:
 status['state']='failed';save();traceback.print_exc()
finally:
 with zipfile.ZipFile(root.parent/(root.name+'-results.zip'),'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
  for p in out.iterdir():z.write(p,'results/'+p.name)
  for p in (root/'manifest.json',root/'run_round4.py'):z.write(p,p.name)
