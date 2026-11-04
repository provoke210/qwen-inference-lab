import json,subprocess,time,zipfile,traceback
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=True)
python='/root/autodl-tmp/qil-env/bin/python';model='/root/autodl-tmp/models/Qwen3-0.6B';workload=str(root/'workload.json')
status={'state':'waiting_for_prior_run','steps':[]}
def save():(out/'status.json').write_text(json.dumps(status,indent=2))
def run(name,args):
 status['current']=name;save()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run([python,*args],cwd=root,stdout=f,stderr=subprocess.STDOUT)
 status['steps'].append(dict(name=name,returncode=r.returncode));save();print(name,r.returncode,flush=True)
 return r.returncode==0
try:
 save()
 previous=root.parent/'qil-paged-20260920-r4b/results/status.json'
 for _ in range(120):
  try:
   if json.loads(previous.read_text())['state'] in ('completed','failed'):break
  except (OSError,ValueError):pass
  time.sleep(5)
 else:raise RuntimeError('prior GPU run did not finish')
 status['state']='running';save()
 run('same-input-probe',['-m','benchmarks.diagnose_paged','--model',model,'--workload',workload,'--output',str(out/'same-input-probe.json')])
 accepted=True
 for group,lengths in [('short',[17,41,83,128]),('long',[63,127,255,511])]:
  name='bfloat16-paged-math-'+group
  ok=run(name,['-m','benchmarks.validate_batched','--model',model,'--workload',workload,'--dtype','bfloat16',
   '--reference-attention','sdpa','--sdpa-backend','math','--reference-cache','--decode-attention','paged',
   '--prompt-lengths',*map(str,lengths),'--output',str(out/(name+'.json'))])
  accepted=accepted and ok
 if accepted:
  run('bfloat16-math-profile',['-m','benchmarks.profile_paged','--model',model,'--workload',workload,'--dtype','bfloat16','--sdpa-backend','math','--output',str(out/'bfloat16-math-profile.json')])
 status['state']='completed';status['accepted']=accepted;save()
except Exception:status['state']='failed';save();traceback.print_exc()
finally:
 with zipfile.ZipFile(root.parent/(root.name+'-results.zip'),'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
  for p in out.iterdir():z.write(p,'results/'+p.name)
  for p in (root/'manifest.json',Path(__file__)):z.write(p,p.name)
