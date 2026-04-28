import argparse
import json
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pfr_utils import build_vocab, encode_tokens, load_tokens_from_file, load_vocab
from train_utils import TrainLogger, set_seed


class CbowDataset(Dataset):
    def __init__(self, ids: List[int], window: int) -> None:
        self.ids = ids
        self.window = window

    def __len__(self) -> int:
        return max(len(self.ids) - 2 * self.window, 0)

    def __getitem__(self, idx: int):
        center = idx + self.window
        context = self.ids[center - self.window:center] + self.ids[center + 1:center + self.window + 1]
        target = self.ids[center]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class CbowModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(context_ids)
        pooled = emb.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--tokens", default="data/processed/pfr_tokens.txt")
    parser.add_argument("--vocab_path", default="data/processed/pfr_vocab.json")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--result_dir", default="result")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    set_seed(args.seed)

    tokens = load_tokens_from_file(args.tokens)
    if os.path.exists(args.vocab_path):
        itos, stoi = load_vocab(args.vocab_path)
    else:
        itos, stoi = build_vocab(tokens, args.vocab_size)

    ids = encode_tokens(tokens, stoi)

    dataset = CbowDataset(ids, args.window)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CbowModel(len(itos), args.embed_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    logger = TrainLogger(
        result_dir=args.result_dir,
        model_name="fnn",
        run_name=args.run_name,
        config=vars(args),
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for context_ids, target in loader:
            context_ids = context_ids.to(device)
            target = target.to(device)
            logits = model(context_ids)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch=epoch + 1, epochs=args.epochs, loss=avg_loss, lr=lr)

    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(args.result_dir, "fnn_embeddings.pt")
    torch.save({"embedding": model.embed.weight.detach().cpu()}, out_path)
    logger.save_summary(extra={"embedding_path": out_path, "vocab_size": len(itos)})

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
