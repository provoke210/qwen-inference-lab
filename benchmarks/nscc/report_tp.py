"""Pull NSCC tensor-parallel results and render a Markdown report for VS Code."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(sys.argv[1])
LOCAL = REPO / 'benchmarks/results/2026-11-19-a100-tp'
REMOTE = os.environ['QIL_REMOTE'] + '/results-tp/'
LOCAL.mkdir(parents=True, exist_ok=True)
if '--no-sync' not in sys.argv:
    subprocess.run(['rsync', '-a', '--delete', '--exclude', 'REPORT.md', REMOTE, str(LOCAL) + '/'], check=False)


def load(name):
    try:
        return json.loads((LOCAL / name).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def mark(ok):
    return '✅' if ok else '❌'


lines = ['# NSCC A100 多卡（张量并行）验证报告', '',
         f'更新时间：{datetime.now():%Y-%m-%d %H:%M:%S}（本文件由脚本生成，会被覆盖）', '']
env = LOCAL / 'environment.txt'
if env.exists():
    lines += ['```text', env.read_text().strip(), '```', '']
current = (LOCAL / 'current.txt').read_text().strip() if (LOCAL / 'current.txt').exists() else '尚未开始（排队中）'
lines += ['## 进度', '', f'当前状态：**{current}**', '']
status = LOCAL / 'status.tsv'
rows = [r.split('\t') for r in status.read_text().splitlines()[1:]] if status.exists() else []
if rows:
    lines += ['| 步骤 | 结果 | 耗时(s) |', '|---|---|---:|']
    lines += [f'| {s} | {mark(c == "0")} {"" if c == "0" else "退出码 " + c} | {sec} |' for s, _, c, sec in rows]
    lines.append('')

def tests_table(name, title):
    xml = LOCAL / name
    if not xml.exists():
        return []
    suite = ElementTree.parse(xml).getroot()
    suite = suite if suite.tag == 'testsuite' else suite.find('testsuite')
    a = {k: int(suite.get(k, 0)) for k in ('tests', 'failures', 'errors', 'skipped')}
    bad = a['failures'] + a['errors']
    failed = [c.get('name') for c in suite.iter('testcase') if c.find('failure') is not None or c.find('error') is not None]
    out = [f'| {title} | {a["tests"]} | {a["tests"] - bad - a["skipped"]} | {bad} | {a["skipped"]} |']
    return out, failed


tables = [(t, tests_table(n, t)) for n, t in (('pytest.xml', '修复前'), ('pytest-fixed.xml', '修复 Transformers 5 兼容后'))]
tables = [(t, r) for t, r in tables if r]
if tables:
    lines += ['## 1. 新环境（torch 2.13 + Transformers 5）下的全部测试', '',
              '| 代码 | 总数 | 通过 | 失败 | 跳过 |', '|---|---:|---:|---:|---:|']
    for title, (row, failed) in tables:
        lines += row
    lines.append('')
    for title, (row, failed) in tables:
        if failed:
            lines += [f'{title}失败用例（前 15 个）：'] + [f'- `{n}`' for n in failed[:15]] + ['']
imp = LOCAL / 'vllm-import-fixed.log'
if imp.exists():
    text = imp.read_text().strip().splitlines()
    lines += ['替换 torchcodec 后在 GPU 节点导入 vLLM：', '', '```text', *text[-3:], '```', '']

tp_files = sorted(LOCAL.glob('tp-*.json'))
if tp_files:
    lines += ['## 2. 1 卡 vs 2 卡（同一份代码，2 卡输出须与 1 卡逐 token 一致）', '',
              '| 模型 / 注意力 | 负载 | 输出 token | 1 卡 (tok/s) | 2 卡 (tok/s) | 2 卡 / 1 卡 | 2 卡输出一致 |',
              '|---|---|---:|---:|---:|---:|:-:|']
    for path in tp_files:
        r = load(path.name)
        label = path.stem.removeprefix('tp-')
        for p in (r or {}).get('performance', []):
            one, two = p['by_size'].get('1'), p['by_size'].get('2')
            t1 = one and one['output_tokens_per_s']
            t2 = two and two['output_tokens_per_s']
            ratio = f'{t2 / t1:.2f}×' if t1 and t2 else '—'
            match = mark(two['matches_single_gpu']) if two else '—'
            lines.append(f"| {label} | {p['name']} | {p['new_tokens']} | {t1 and f'{t1:.1f}' or '—'} "
                         f"| {t2 and f'{t2:.1f}' or '—'} | {ratio} | {match} |")
    lines.append('')

qil, vllm = load('qil-single-fp32.json'), load('vllm-fp32-eager.json')
if qil or vllm:
    lines += ['## 3. 单卡 qil vs vLLM 0.30.0（Qwen3-0.6B，FP32，eager，前缀缓存关闭）', '',
              '| 负载 | qil (tok/s) | vLLM (tok/s) | qil / vLLM |', '|---|---:|---:|---:|']
    pick = lambda r, n: next((p for p in (r or {}).get('performance', []) if p['name'] == n), None)
    for name in [p['name'] for p in (qil or vllm).get('performance', [])]:
        q, v = pick(qil, name), pick(vllm, name)
        qt, vt = q and q['output_tokens_per_s'], v and v['output_tokens_per_s']
        lines.append(f"| {name} | {qt and f'{qt:.1f}' or '—'} | {vt and f'{vt:.1f}' or '—'} "
                     f"| {f'{qt / vt * 100:.1f}%' if qt and vt else '—'} |")
    lines.append('')

lines += ['## 原始文件', '', '本目录下的 `*.log` 是每一步的完整输出，`*.json` 是原始测量数据。', '']
(LOCAL / 'REPORT.md').write_text('\n'.join(lines))
print(current)
