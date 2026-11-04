import json,os,subprocess,time,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'results'
for attempt in range(180):
    state=json.loads((out/'status.json').read_text())['state']
    if state!='running': break
    time.sleep(5)
assert state=='completed',state
w=json.loads((root/'workload.json').read_text())
w['performance']=[c for c in w['performance'] if c['name']=='input512_batch4']
(out/'repeat-workload.json').write_text(json.dumps(w))
env=dict(os.environ,PYTHONPATH=str(root),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
cmd=['/root/autodl-tmp/qil-env/bin/python','-m','benchmarks.profile_prefill','--model','/root/autodl-tmp/models/Qwen3-0.6B','--workload',str(out/'repeat-workload.json'),'--output',str(out/'repeat.json'),'--runs','12']
print('START REPEAT',flush=True)
subprocess.run(cmd,cwd=root,env=env,check=True,timeout=600)
(out/'repeat_check.py').write_text(Path(__file__).read_text())
with zipfile.ZipFile(root.parent/(root.name+'-complete-results.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.iterdir():
        if p.is_file(): z.write(p,p.name)
print('REPEAT_AND_ARCHIVE_COMPLETE',flush=True)
