"""Run isolated correctness and comparison processes in the existing GPU envs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu-results'
OUT.mkdir(exist_ok=True)
QIL = '/root/autodl-tmp/qil-env/bin/python'
VLLM = '/root/autodl-tmp/vllm-env/bin/python'
MODEL = '/root/autodl-tmp/models/Qwen3-0.6B'
WORK = OUT / 'workload.json'
STATUS = dict(started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), state='running', steps=[])
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')


def save_status():
    (OUT / 'status.json').write_text(json.dumps(STATUS, indent=2) + '\n')


def run(label, command):
    STATUS['current_step'] = label
    save_status()
    print('START ' + label, flush=True)
    start = time.monotonic()
    with (OUT / (label + '.log')).open('w') as log:
        result = subprocess.run([str(x) for x in command], cwd=ROOT, env=ENV,
                                stdout=log, stderr=subprocess.STDOUT, timeout=900)
    STATUS['steps'].append(dict(label=label, returncode=result.returncode,
                                elapsed_s=round(time.monotonic()-start, 3)))
    save_status()
    print('DONE ' + label + ' returncode=' + str(result.returncode), flush=True)
    if result.returncode:
        raise RuntimeError(label + ' failed; see ' + label + '.log')


def compact(dtype):
    reports = {mode: json.loads((OUT / f'{dtype}-{mode}.json').read_text())
               for mode in ('sequential', 'batched', 'vllm')}
    baseline, batched, vllm = [reports[mode] for mode in ('sequential', 'batched', 'vllm')]
    for report in reports.values():
        assert report['complete']
        for field in ('dtype', 'gpu', 'torch', 'model', 'workload_sha256', 'runs', 'warmups'):
            assert report[field] == baseline[field], field
    quality = []
    for a, b, c in zip(baseline['quality'], batched['quality'], vllm['quality']):
        quality.append(dict(name=a['name'], sequential_equals_batched=a['token_ids']==b['token_ids'],
                            batched_equals_vllm=b['token_ids']==c['token_ids']))
    cases = []
    for a, b, c in zip(baseline['performance'], batched['performance'], vllm['performance']):
        assert a['name'] == b['name'] == c['name']
        cases.append(dict(name=a['name'], batch=a['batch'], prompt_tokens=a['prompt_tokens'],
            sequential_tokens_per_s=a['output_tokens_per_s'], batched_tokens_per_s=b['output_tokens_per_s'],
            vllm_tokens_per_s=c['output_tokens_per_s'], speedup=a['median_s']/b['median_s'],
            vllm_percentage=100*b['output_tokens_per_s']/c['output_tokens_per_s'],
            sequential_equals_batched=a['output_token_ids_by_run']==b['output_token_ids_by_run'],
            batched_equals_vllm=b['output_token_ids_by_run']==c['output_token_ids_by_run']))
    return dict(dtype=dtype, quality=quality, performance=cases)


try:
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    run('environment', [QIL, '-c',
        'import torch,transformers,json,platform; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),memory_bytes=torch.cuda.get_device_properties(0).total_memory)))'])
    run('prepare', [QIL, '-m', 'benchmarks.compare_independent', '--prepare', '--model', MODEL, '--workload', WORK])
    summary = []
    for dtype in ('float32', 'bfloat16'):
        run(dtype + '-correctness', [QIL, '-m', 'benchmarks.validate_batched', '--model', MODEL,
            '--workload', WORK, '--dtype', dtype, '--output', OUT / (dtype + '-correctness.json')])
        for mode, backend, python in [('sequential', 'qil', QIL), ('batched', 'qil', QIL), ('vllm', 'vllm', VLLM)]:
            cmd = [python, '-m', 'benchmarks.compare_independent', '--backend', backend, '--model', MODEL,
                   '--workload', WORK, '--dtype', dtype, '--runs', '5', '--output', OUT / f'{dtype}-{mode}.json']
            if mode == 'batched':
                cmd.append('--batch-decode')
            run(dtype + '-' + mode, cmd)
        summary.append(compact(dtype))
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    STATUS['state'] = 'completed'
except Exception as error:
    STATUS['state'] = 'failed'
    STATUS['error'] = str(error)
finally:
    STATUS['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    save_status()
    archive = ROOT.parent / (ROOT.name + '-results.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(OUT.glob('*')):
            if path.is_file():
                output.write(path, path.name)
    print(json.dumps(dict(state=STATUS['state'], results_archive=str(archive), error=STATUS.get('error'))), flush=True)
if STATUS['state'] != 'completed':
    sys.exit(1)
