"""Bounded FP32 validation and profiling on the existing GPU instance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
QIL = '/root/autodl-tmp/qil-env/bin/python'
VLLM = '/root/autodl-tmp/vllm-env/bin/python'
MODEL = '/root/autodl-tmp/models/Qwen3-0.6B'
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
status = dict(state='running', started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), steps=[])

def save():
    (OUT/'status.json').write_text(json.dumps(status, indent=2)+'\n')

def run(name, cmd):
    status['current_step'] = name
    save()
    print('START', name, flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        result = subprocess.run([str(s) for s in cmd], cwd=ROOT, env=ENV,
                                stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    status['steps'].append(dict(name=name, returncode=result.returncode))
    save()
    print('DONE', name, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(name+' failed')

try:
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    run('environment', [QIL, '-c', 'import torch,platform,json;print(json.dumps(dict(torch=torch.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(),memory_bytes=torch.cuda.get_device_properties(0).total_memory)))'])
    run('correctness', [QIL, '-m', 'benchmarks.validate_batched', '--model', MODEL,
                       '--workload', ROOT/'workload.json', '--output', OUT/'correctness.json', '--dtype', 'float32'])
    run('paired', [QIL, '-m', 'benchmarks.profile_prefill', '--model', MODEL,
                   '--workload', ROOT/'extended-workload.json', '--output', OUT/'paired.json', '--runs', '7', '--profile'])
    for name, python, backend, flags in [('qil',QIL,'qil',['--batch-decode','--selective-logits']),('vllm',VLLM,'vllm',[])]:
        run(name,[python,'-m','benchmarks.compare_independent','--model',MODEL,'--workload',ROOT/'workload.json',
                  '--backend',backend,'--dtype','float32','--runs','5','--output',OUT/(name+'.json'),*flags])
    q, v = [json.loads((OUT/(name+'.json')).read_text()) for name in ('qil','vllm')]
    assert q['complete'] and v['complete']
    assert all(a['token_ids']==b['token_ids'] for a,b in zip(q['quality'],v['quality']))
    assert all(a['output_token_ids_by_run']==b['output_token_ids_by_run'] for a,b in zip(q['performance'],v['performance']))
    status['state'] = 'completed'
    status['qil_vllm_tokens_exact'] = True
except Exception as error:
    status['state']='failed'
    status['error']=repr(error)
    raise
finally:
    status['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S %z')
    save()
    archive=ROOT.parent/(ROOT.name+'-results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in OUT.iterdir():
            if path.is_file():
                bundle.write(path,path.name)
    print('ARCHIVE',archive,flush=True)
