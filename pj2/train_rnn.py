import argparse
import json
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pfr_utils import build_vocab, encode_tokens, load_tokens_from_file, load_vocab
from train_utils import TrainLogger, set_seed


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
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        emb = self.emb_dropout(emb)
        out, _ = self.rnn(emb)
        out = self.out_dropout(out)
        return self.fc(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--tokens", default="data/processed/pfr_tokens.txt")
    parser.add_argument("--vocab_path", default="data/processed/pfr_vocab.json")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
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

    dataset = LMDataset(ids, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RnnLM(
        len(itos),
        args.embed_dim,
        args.hidden_size,
        args.num_layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    logger = TrainLogger(
        result_dir=args.result_dir,
        model_name="rnn",
        run_name=args.run_name,
        config=vars(args),
    )

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
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch=epoch + 1, epochs=args.epochs, loss=avg_loss, lr=lr)

    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(args.result_dir, "rnn_embeddings.pt")
    torch.save({"embedding": model.embed.weight.detach().cpu()}, out_path)
    logger.save_summary(extra={"embedding_path": out_path, "vocab_size": len(itos)})

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
