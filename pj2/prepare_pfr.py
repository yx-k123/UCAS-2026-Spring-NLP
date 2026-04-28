import argparse
import json
import os
import re
from typing import List

from pfr_utils import build_vocab, load_tokens, save_tokens, save_vocab


def load_english_tokens_from_json(json_path: str, field: str = "content") -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list, got: {type(data)}")

    tokens: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get(field, "")).lower()
        tokens.extend(re.findall(r"[a-z]+", text))
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="data/raw/PeopleDaily199801.txt")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh")
    parser.add_argument("--json_field", default="content")
    parser.add_argument("--tokens_out", default="data/processed/pfr_tokens.txt")
    parser.add_argument("--vocab_out", default="data/processed/pfr_vocab.json")
    parser.add_argument("--vocab_size", type=int, default=1000)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    if args.lang == "zh":
        tokens = load_tokens(args.input)
    else:
        tokens = load_english_tokens_from_json(args.input, field=args.json_field)
    save_tokens(tokens, args.tokens_out)

    itos, _ = build_vocab(tokens, args.vocab_size)
    save_vocab(itos, args.vocab_out)

    print(f"tokens: {len(tokens)}")
    print(f"vocab_size: {len(itos)}")
    print(f"tokens_out: {os.path.abspath(args.tokens_out)}")
    print(f"vocab_out: {os.path.abspath(args.vocab_out)}")


if __name__ == "__main__":
    main()
