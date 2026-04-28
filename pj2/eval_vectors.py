import argparse
import json
import os
import random
from typing import List, Tuple

import torch


def load_vocab(vocab_path: str) -> Tuple[List[str], dict]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    itos = data["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def load_embedding(emb_path: str) -> torch.Tensor:
    data = torch.load(emb_path, map_location="cpu")
    return data["embedding"]


def top_k_similar(emb: torch.Tensor, idx: int, k: int) -> List[int]:
    vec = emb[idx]
    vec = vec / (vec.norm() + 1e-9)
    norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-9)
    sims = torch.matmul(norm, vec)
    sims[idx] = -1.0
    topk = torch.topk(sims, k=k).indices.tolist()
    return topk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fnn", "rnn", "lstm"], default="fnn")
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--vocab_path", default="data/processed/pfr_vocab.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emb_path = os.path.join(args.result_dir, f"{args.model}_embeddings.pt")
    emb = load_embedding(emb_path)
    itos, _ = load_vocab(args.vocab_path)

    random.seed(args.seed)
    candidates = list(range(2, min(len(itos), emb.size(0))))
    picks = random.sample(candidates, k=min(args.sample, len(candidates)))

    out_path = os.path.join(args.result_dir, f"similar_{args.model}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in picks:
            word = itos[idx]
            top_ids = top_k_similar(emb, idx, args.k)
            top_words = [itos[i] for i in top_ids]
            line = word + "\t" + ",".join(top_words)
            f.write(line + "\n")

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
