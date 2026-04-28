import argparse
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.font_manager as fm

noto_fonts = [f for f in fm.findSystemFonts() if 'NotoSansCJK' in f]
if noto_fonts:
    fm.fontManager.addfont(noto_fonts[0])
    for font in fm.fontManager.ttflist:
        if font.fname == noto_fonts[0]:
            actual_family = font.name
            break
    else:
        actual_family = "Noto Sans CJK SC"
    matplotlib.rcParams["font.family"] = actual_family
else:
    print("Warning: Noto Sans CJK font not found, Chinese may not display correctly.")

def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        itos = json.load(f)["itos"]
    return itos

def load_embedding(model: str, result_dir: str):
    emb_path = os.path.join(result_dir, f"{model}_embeddings.pt")
    data = torch.load(emb_path, map_location="cpu")
    return data["embedding"]

def write_tensorboard(emb, itos, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)
    writer.add_embedding(emb, metadata=itos)
    writer.close()
    print(f"tensorboard logdir: {os.path.abspath(out_dir)}")

def plot_2d(emb, itos, method, n, out_png):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = min(n, emb.size(0))
    vecs = emb[:n].numpy()
    labels = itos[:n]

    if method == "pca":
        reducer = PCA(n_components=2)
        xy = reducer.fit_transform(vecs)
    else:
        reducer = TSNE(n_components=2, init="random", random_state=42, perplexity=30)
        xy = reducer.fit_transform(vecs)

    plt.figure(figsize=(12, 10))
    plt.scatter(xy[:, 0], xy[:, 1], s=10)
    for i, w in enumerate(labels):
        plt.text(xy[i, 0], xy[i, 1], w, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"saved: {os.path.abspath(out_png)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fnn", "rnn", "lstm"], default="fnn")
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--vocab_path", default="data/processed/pfr_vocab.json")
    parser.add_argument("--tb_dir", default="result/tb")
    parser.add_argument("--plot", choices=["none", "pca", "tsne"], default="none")
    parser.add_argument("--plot_n", type=int, default=200)
    parser.add_argument("--plot_out", default="result/emb_2d.png")
    args = parser.parse_args()

    itos = load_vocab(args.vocab_path)
    emb = load_embedding(args.model, args.result_dir)

    write_tensorboard(emb, itos, args.tb_dir)

    if args.plot != "none":
        plot_2d(emb, itos, args.plot, args.plot_n, args.plot_out)

if __name__ == "__main__":
    main()