import argparse
import json
import os

from pfr_utils import build_vocab, load_tokens, save_tokens, save_vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="data/raw/PeopleDaily199801.txt")
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

    tokens = load_tokens(args.input)
    save_tokens(tokens, args.tokens_out)

    itos, _ = build_vocab(tokens, args.vocab_size)
    save_vocab(itos, args.vocab_out)

    print(f"tokens: {len(tokens)}")
    print(f"vocab_size: {len(itos)}")
    print(f"tokens_out: {os.path.abspath(args.tokens_out)}")
    print(f"vocab_out: {os.path.abspath(args.vocab_out)}")


if __name__ == "__main__":
    main()
