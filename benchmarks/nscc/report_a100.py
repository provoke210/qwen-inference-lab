"""Pull NSCC results and render a Markdown status report for viewing in the editor."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(sys.argv[1])
LOCAL = REPO / 'benchmarks/results/2026-11-16-a100'
REMOTE = os.environ['QIL_REMOTE'] + '/results/'
LOCAL.mkdir(parents=True, exist_ok=True)
if '--no-sync' not in sys.argv:
    subprocess.run(['rsync', '-a', '--delete', '--exclude', 'REPORT.md', REMOTE, str(LOCAL) + '/'], check=True)


def load(name):
    path = LOCAL / name
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None


def mark(ok):
    return '✅' if ok else '❌'


lines = ['# NSCC A100 验证报告', '',
         f'更新时间：{datetime.now():%Y-%m-%d %H:%M:%S}（本文件由脚本生成，会被覆盖）', '']
env = LOCAL / 'environment.txt'
if env.exists():
    lines += ['```text', env.read_text().strip(), '```', '']
lines += ['`base` = 改动前（git HEAD），`new` = 工作区的改动。两者在同一个作业、同一张 GPU 上先后运行。', '']

current = (LOCAL / 'current.txt').read_text().strip() if (LOCAL / 'current.txt').exists() else '尚未开始（排队中）'
lines += ['## 进度', '', f'当前状态：**{current}**', '']
status = LOCAL / 'status.tsv'
rows = [r.split('\t') for r in status.read_text().splitlines()[1:]] if status.exists() else []
if rows:
    lines += ['| 步骤 | 代码 | 结果 | 耗时(s) |', '|---|---|---|---:|']
    lines += [f'| {s} | {t} | {mark(c == "0")} {"" if c == "0" else "退出码 " + c} | {sec} |' for s, t, c, sec in rows]
    lines.append('')

# ---- tests
tests = []
for tree in ('base', 'new'):
    path = LOCAL / f'pytest.{tree}.xml'
    if path.exists():
        suite = ElementTree.parse(path).getroot()
        suite = suite if suite.tag == 'testsuite' else suite.find('testsuite')
        a = {k: int(suite.get(k, 0)) for k in ('tests', 'failures', 'errors', 'skipped')}
        failed = [c.get('name') for c in suite.iter('testcase') if c.find('failure') is not None or c.find('error') is not None]
        tests.append((tree, a, failed))
if tests:
    lines += ['## 1. 测试（含 CUDA 上的 Triton 内核用例）', '', '| 代码 | 总数 | 通过 | 失败 | 跳过 |', '|---|---:|---:|---:|---:|']
    for tree, a, failed in tests:
        bad = a['failures'] + a['errors']
        lines.append(f"| {tree} | {a['tests']} | {a['tests'] - bad - a['skipped']} | {bad} | {a['skipped']} |")
    lines.append('')
    for tree, a, failed in tests:
        if failed:
            lines += [f'{tree} 失败用例：'] + [f'- `{n}`' for n in failed] + ['']

# ---- compile count
compile_ = {t: load(f'compile.{t}.json') for t in ('base', 'new')}
if any(compile_.values()):
    lines += ['## 2. Paged 内核编译次数（单请求，32 → 1032 token，空缓存冷启动）', '',
              '| 指标 | base | new |', '|---|---:|---:|']
    def cell(tree, fn):
        r = compile_[tree]
        try:
            return fn(r) if r else '—'
        except (KeyError, TypeError):
            return '—'
    for label, fn in [('编译出的内核变体数', lambda r: str(r['compiled_kernel_variants'])),
                      ('冷启动总耗时 (s)', lambda r: f"{r['cold']['total_s']:.2f}"),
                      ('热启动总耗时 (s)', lambda r: f"{r['warm']['total_s']:.2f}"),
                      ('冷启动卡顿步数', lambda r: str(r['cold']['stall_steps'])),
                      ('卡顿累计耗时 (s)', lambda r: f"{r['cold']['stall_total_s']:.2f}"),
                      ('冷启动最慢一步 (ms)', lambda r: f"{r['cold']['decode_max_ms']:.1f}"),
                      ('热启动 decode 中位数 (ms)', lambda r: f"{r['warm']['decode_median_ms']:.2f}")]:
        lines.append(f'| {label} | {cell("base", fn)} | {cell("new", fn)} |')
    if all(compile_.values()):
        same = compile_['base']['token_ids'] == compile_['new']['token_ids']
        lines += ['', f'base 与 new 的 1000 个输出 token 完全一致：{mark(same)}']
    lines.append('')

# ---- end-to-end performance
perf_rows = []
for attn in ('dense', 'paged'):
    base, new = load(f'perf-fp32-{attn}.base.json'), load(f'perf-fp32-{attn}.new.json')
    if not (base or new):
        continue
    names = [p['name'] for p in (new or base).get('performance', [])]
    for name in names:
        pick = lambda r: next((p for p in r.get('performance', []) if p['name'] == name), None) if r else None
        b, n = pick(base), pick(new)
        bt, nt = (b or {}).get('output_tokens_per_s'), (n or {}).get('output_tokens_per_s')
        delta = f'{(nt / bt - 1) * 100:+.1f}%' if bt and nt else '—'
        keys = [k for k in (b or {}) if 'token_ids' in k or 'outputs' in k]
        same = '—'
        if b and n and keys:
            same = mark(all(b[k] == n.get(k) for k in keys))
        perf_rows.append(f"| {attn} | {name} | {bt and f'{bt:.1f}' or '—'} | {nt and f'{nt:.1f}' or '—'} | {delta} | {same} |")
if perf_rows:
    lines += ['## 3. 端到端吞吐（FP32，每请求 32 个输出 token，5 次取中位数）', '',
              '| 注意力 | 负载 | base (tok/s) | new (tok/s) | 变化 | 输出一致 |', '|---|---|---:|---:|---:|:-:|'] + perf_rows + ['']

# ---- acceptance against Transformers
accept = sorted(LOCAL.glob('accept-*.json'))
if accept:
    lines += ['## 4. 真实模型数值验收（对 Transformers 4.51.3 参考）', '',
              '配置名 = 精度-注意力-SDPA 后端-长度组。“未运行”表示参考实现在该后端下无可用 kernel，脚本提前退出。', '',
              '| 配置 | 代码 | 总体 | logits 在容差内 | 最大 logit 误差 | argmax 一致 | 生成逐 token 一致 | 页面全部回收 |',
              '|---|---|:-:|---:|---:|---:|:-:|:-:|']
    exits = {(s, t): c for s, t, c, _ in rows}
    for path in accept:
        r = load(path.name)
        config, tree = path.name.removesuffix('.json').rsplit('.', 1)
        if not r or not r.get('logit_checks'):
            lines.append(f'| {config.removeprefix("accept-")} | {tree} | — | 未运行 | — | — | — | — |')
            continue
        checks, gens = r['logit_checks'], r.get('generations', [])
        exact = bool(gens) and all(all(g.get('exact_match', [])) for g in gens)
        pages = bool(gens) and all(g.get('all_pages_released') for g in gens)
        lines.append(f"| {config.removeprefix('accept-')} | {tree} | {mark(exits.get((config, tree)) == '0')} "
                     f"| {sum(c['within_tolerance'] for c in checks)}/{len(checks)} "
                     f"| {max(c['max_abs_error'] for c in checks):.3g} "
                     f"| {sum(c['argmax_match'] for c in checks)}/{len(checks)} | {mark(exact)} | {mark(pages)} |")
    missing = sorted((s, t) for (s, t), c in exits.items() if s.startswith('accept-') and c != '0'
                     and not (LOCAL / f'{s}.{t}.json').exists())
    for s_, t_ in missing:
        lines.append(f'| {s_.removeprefix("accept-")} | {t_} | ❌ | 未运行 | — | — | — | — |')
    lines.append('')

# ---- paired dense/grouped/paged profile
profile = {t: load(f'profile-paged-fp32.{t}.json') for t in ('base', 'new')}
if any(profile.values()):
    lines += ['## 5. dense / grouped / paged 配对测量（FP32，tok/s）', '',
              '| 负载 | 代码 | dense | grouped | paged | paged 相对 dense | 三者输出一致 |', '|---|---|---:|---:|---:|---:|:-:|']
    for tree, r in profile.items():
        for p in (r or {}).get('performance', []):
            t = p.get('output_tokens_per_s', {})
            gain = f"{(t['paged'] / t['dense'] - 1) * 100:+.1f}%" if t.get('paged') and t.get('dense') else '—'
            lines.append(f"| {p['name']} | {tree} | {t.get('dense', 0):.1f} | {t.get('grouped', 0):.1f} "
                         f"| {t.get('paged', 0):.1f} | {gain} | {mark(p.get('all_outputs_match'))} |")
    lines.append('')

lines += ['## 原始文件', '', '本目录下的 `*.log` 是每一步的完整输出，`*.json` 是原始测量数据。', '']
(LOCAL / 'REPORT.md').write_text('\n'.join(lines))
print(current, '|', len(rows), 'steps recorded')
