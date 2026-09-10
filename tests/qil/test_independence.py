"""A real checkpoint can run with all legacy/model-framework imports blocked."""
import subprocess
import sys


def test_runtime_does_not_import_legacy_or_transformers(pair, tmp_path):
    pair[1].save_pretrained(tmp_path)
    code = '''
import builtins, sys
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'nano_vllm', 'nanovllm', 'vllm', 'transformers', 'triton', 'flash_attn'}:
        raise AssertionError('Forbidden runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
from qil import Qwen3, Engine
engine = Engine(Qwen3.from_directory(sys.argv[1]), num_pages=8, page_size=4)
result = engine.generate([[1, 2, 3]], max_new_tokens=2, ignore_eos=True)[0]
assert len(result['token_ids']) == 2 and result['state'] == 'completed'
assert not engine.pool.tables
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True, timeout=60)
