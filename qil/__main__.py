"""python -m qil --model DIR --prompt 'Hello' --device cpu"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--token-ids', help='JSON list; runs without Transformers/tokenizer dependency')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--sdpa-backend', default='auto', choices=['auto', 'math', 'efficient'])
    parser.add_argument('--max-new-tokens', default=64, type=int)
    parser.add_argument('--pages', default=256, type=int)
    parser.add_argument('--page-size', default=16, type=int)
    parser.add_argument('--chunk-size', default=64, type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--decode-attention', choices=['dense', 'grouped', 'paged'], default='dense')
    parser.add_argument('--tensor-parallel', default=1, type=int, metavar='N',
                        help='split the model over N GPUs (one process each); cuda only')
    args = parser.parse_args()
    if args.tensor_parallel > 1 and (args.device != 'cuda' or args.sdpa_backend != 'auto'):
        parser.error('--tensor-parallel requires --device cuda and the default --sdpa-backend')
    import torch
    from .engine import Engine
    from .model import Qwen3
    tokenizer = None
    if args.prompt is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        tokens = json.loads(args.token_ids)
    if args.tensor_parallel > 1:
        from . import parallel
        result = parallel.generate(args.model, [tokens], tp_size=args.tensor_parallel, dtype=args.dtype,
                                   max_new_tokens=args.max_new_tokens, num_pages=args.pages,
                                   page_size=args.page_size, prefill_chunk=args.chunk_size,
                                   decode_attention=args.decode_attention)['results'][0]
        return report(result, tokenizer, args.output)
    model = Qwen3.from_directory(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    engine = Engine(model, num_pages=args.pages, page_size=args.page_size, prefill_chunk=args.chunk_size, decode_attention=args.decode_attention)
    try:
        from contextlib import nullcontext
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backends = {'math': SDPBackend.MATH, 'efficient': SDPBackend.EFFICIENT_ATTENTION}
        context = nullcontext() if args.sdpa_backend == 'auto' else sdpa_kernel(backends[args.sdpa_backend])
        with context:
            result = engine.generate([tokens], max_new_tokens=args.max_new_tokens)[0]
        report(result, tokenizer, args.output)
    finally:
        engine.close()


def report(result, tokenizer, output):
    if tokenizer:
        result['text'] = tokenizer.decode(result['token_ids'], skip_special_tokens=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
