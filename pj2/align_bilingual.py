import argparse
import json
import os
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
    return data["embedding"].float()


def parse_lexicon(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                src, tgt = line.split("\t", 1)
            elif "," in line:
                src, tgt = line.split(",", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                src, tgt = parts[0], parts[1]
            pairs.append((src.strip(), tgt.strip()))
    return pairs


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-9)


def topk_tgt(src_vec: torch.Tensor, tgt_emb: torch.Tensor, k: int) -> List[int]:
    sims = torch.matmul(tgt_emb, src_vec)
    return torch.topk(sims, k=k).indices.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_vocab", required=True)
    parser.add_argument("--src_emb", required=True)
    parser.add_argument("--tgt_vocab", required=True)
    parser.add_argument("--tgt_emb", required=True)
    parser.add_argument("--lexicon", required=True, help="bilingual seed lexicon: src\\ttgt")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--report_out", default="result/bilingual_alignment_report.txt")
    args = parser.parse_args()

    src_itos, src_stoi = load_vocab(args.src_vocab)
    tgt_itos, tgt_stoi = load_vocab(args.tgt_vocab)
    src_emb = normalize_rows(load_embedding(args.src_emb))
    tgt_emb = normalize_rows(load_embedding(args.tgt_emb))

    lexicon = parse_lexicon(args.lexicon)
    anchors = [(s, t) for s, t in lexicon if s in src_stoi and t in tgt_stoi]
    if len(anchors) < 10:
        raise ValueError(f"Too few valid lexicon pairs: {len(anchors)} (need >= 10)")

    src_idx = [src_stoi[s] for s, _ in anchors]
    tgt_idx = [tgt_stoi[t] for _, t in anchors]
    x = src_emb[src_idx]
    y = tgt_emb[tgt_idx]

    m = torch.matmul(x.T, y)
    u, _, v_t = torch.linalg.svd(m)
    w = torch.matmul(u, v_t)
    src_aligned = torch.matmul(src_emb, w)

    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(f"valid_anchors\t{len(anchors)}\n")
        f.write("src_word\ttgt_word\tcosine\tl2\n")
        for s, t in anchors:
            s_idx = src_stoi[s]
            t_idx = tgt_stoi[t]
            s_vec = src_aligned[s_idx]
            t_vec = tgt_emb[t_idx]
            cosine = torch.dot(s_vec, t_vec).item()
            l2 = torch.norm(s_vec - t_vec).item()
            f.write(f"{s}\t{t}\t{cosine:.6f}\t{l2:.6f}\n")

        f.write("\n# top-k target words for each source anchor\n")
        for s, t in anchors:
            s_idx = src_stoi[s]
            top_ids = topk_tgt(src_aligned[s_idx], tgt_emb, args.topk)
            top_words = [tgt_itos[i] for i in top_ids]
            f.write(f"{s}\t{t}\t{','.join(top_words)}\n")

    print(f"saved: {args.report_out}")


if __name__ == "__main__":
    main()
