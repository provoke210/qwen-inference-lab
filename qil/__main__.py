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
    parser.add_argument('--max-new-tokens', default=64, type=int)
    parser.add_argument('--pages', default=256, type=int)
    parser.add_argument('--page-size', default=16, type=int)
    parser.add_argument('--chunk-size', default=64, type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
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
    model = Qwen3.from_directory(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    engine = Engine(model, num_pages=args.pages, page_size=args.page_size, prefill_chunk=args.chunk_size)
    try:
        result = engine.generate([tokens], max_new_tokens=args.max_new_tokens)[0]
        if tokenizer:
            result['text'] = tokenizer.decode(result['token_ids'], skip_special_tokens=True)
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + '\n')
        print(encoded)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
