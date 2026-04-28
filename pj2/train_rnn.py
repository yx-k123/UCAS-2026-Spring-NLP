import argparse
import json
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pfr_utils import build_vocab, encode_tokens, load_tokens_from_file, load_vocab


class LMDataset(Dataset):
    def __init__(self, ids: List[int], seq_len: int) -> None:
        self.ids = ids
        self.seq_len = seq_len
        self.total = max(len(ids) - seq_len, 0)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        seq = self.ids[idx:idx + self.seq_len + 1]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class RnnLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        return self.fc(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--tokens", default="data/processed/pfr_tokens.txt")
    parser.add_argument("--vocab_path", default="data/processed/pfr_vocab.json")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--result_dir", default="result")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    tokens = load_tokens_from_file(args.tokens)
    if os.path.exists(args.vocab_path):
        itos, stoi = load_vocab(args.vocab_path)
    else:
        itos, stoi = build_vocab(tokens, args.vocab_size)

    ids = encode_tokens(tokens, stoi)

    dataset = LMDataset(ids, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RnnLM(len(itos), args.embed_dim, args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f}")

    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(args.result_dir, "rnn_embeddings.pt")
    torch.save({"embedding": model.embed.weight.detach().cpu()}, out_path)

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
