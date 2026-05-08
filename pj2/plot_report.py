import argparse
import json
import os
import random
from glob import glob
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


MODELS = ["fnn", "rnn", "lstm"]


def configure_plot_style() -> None:
    # Report-friendly defaults: consistent fonts, sizes, and spacing.
    matplotlib.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def configure_chinese_font() -> None:
    noto_fonts = [f for f in fm.findSystemFonts() if "NotoSansCJK" in f]
    if noto_fonts:
        fm.fontManager.addfont(noto_fonts[0])
        for font in fm.fontManager.ttflist:
            if font.fname == noto_fonts[0]:
                matplotlib.rcParams["font.family"] = font.name
                break
    matplotlib.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_latest_file(pattern: str) -> str:
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return max(candidates, key=os.path.getmtime)


def load_last_run_losses(jsonl_path: str) -> List[Tuple[int, float]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return []

    # Keep only the final contiguous run.
    start_idx = 0
    for i in range(1, len(rows)):
        if rows[i].get("step", 0) <= rows[i - 1].get("step", 0):
            start_idx = i
    rows = rows[start_idx:]

    out = []
    for r in rows:
        out.append((int(r["epoch"]), float(r["loss"])))
    return out


def load_similar_map(path: str) -> Dict[str, List[str]]:
    data: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w, ns = line.split("\t")
            data[w] = ns.split(",")
    return data


def mean_overlap(sim_a: Dict[str, List[str]], sim_b: Dict[str, List[str]]) -> float:
    words = sorted(set(sim_a.keys()) & set(sim_b.keys()))
    if not words:
        return 0.0
    vals = []
    for w in words:
        vals.append(len(set(sim_a[w]) & set(sim_b[w])) / 10.0)
    return mean(vals)


def parse_alignment_cosines(path: str) -> List[float]:
    cos = []
    in_topk = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# top-k"):
                in_topk = True
                continue
            if in_topk:
                continue
            if line.startswith("valid_anchors") or line.startswith("src_word"):
                continue
            parts = line.split("\t")
            if len(parts) == 4:
                cos.append(float(parts[2]))
    return cos


def fig1_training_curves(result_zh: str, result_en: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, result_dir, title in [
        (axes[0], result_zh, "Chinese Loss Curves"),
        (axes[1], result_en, "English Loss Curves"),
    ]:
        for m in MODELS:
            log_path = pick_latest_file(os.path.join(result_dir, f"train_{m}_*.jsonl"))
            points = load_last_run_losses(log_path)
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=m.upper())
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig2_model_overlap(result_zh: str, result_en: str, out_path: str) -> None:
    pairs = [("fnn", "rnn"), ("fnn", "lstm"), ("rnn", "lstm")]
    labels = ["FNN-RNN", "FNN-LSTM", "RNN-LSTM"]

    def calc(result_dir: str) -> List[float]:
        sims = {
            m: load_similar_map(os.path.join(result_dir, f"similar_{m}.txt"))
            for m in MODELS
        }
        return [mean_overlap(sims[a], sims[b]) for a, b in pairs]

    zh_vals = calc(result_zh)
    en_vals = calc(result_en)

    x = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(
        [i - width / 2 for i in x],
        zh_vals,
        width,
        label="Chinese",
        color="#1f77b4",
        edgecolor="#1f77b4",
        alpha=0.9,
    )
    ax.bar(
        [i + width / 2 for i in x],
        en_vals,
        width,
        label="English",
        color="#ff7f0e",
        edgecolor="#ff7f0e",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Top-10 Overlap")
    ax.set_title("Model Similarity Overlap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig3_alignment_quality(result_align: str, out_path: str) -> None:
    full_cos = parse_alignment_cosines(os.path.join(result_align, "zh_en_alignment.txt"))
    compact_cos = parse_alignment_cosines(
        os.path.join(result_align, "zh_en_alignment_compact.txt")
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.boxplot(
        [full_cos, compact_cos],
        tick_labels=["Full Lexicon", "Compact Lexicon"],
        patch_artist=True,
        boxprops={"facecolor": "#e0ecf8", "edgecolor": "#4a7ebb"},
        medianprops={"color": "#d62728", "linewidth": 1.6},
        whiskerprops={"color": "#4a7ebb"},
        capprops={"color": "#4a7ebb"},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#8da0cb"},
    )
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Zh-En Alignment Quality")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig4_case_table(result_dir: str, out_path: str, sample_n: int, seed: int) -> None:
    sims = {m: load_similar_map(os.path.join(result_dir, f"similar_{m}.txt")) for m in MODELS}
    common_words = sorted(set.intersection(*(set(v.keys()) for v in sims.values())))
    if not common_words:
        raise ValueError("No common words among similar_fnn/rnn/lstm outputs.")

    random.seed(seed)
    picks = random.sample(common_words, k=min(sample_n, len(common_words)))

    rows = []
    for w in picks:
        rows.append(
            [
                w,
                ", ".join(sims["fnn"][w][:5]),
                ", ".join(sims["rnn"][w][:5]),
                ", ".join(sims["lstm"][w][:5]),
            ]
        )

    fig, ax = plt.subplots(figsize=(13.0, max(5.0, 0.36 * len(rows) + 1.4)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Word", "FNN Top-5", "RNN Top-5", "LSTM Top-5"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#fafafa")
    ax.set_title("Case Study: Nearest Neighbors Comparison", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_zh", default="result_zh")
    parser.add_argument("--result_en", default="result_en")
    parser.add_argument("--result_align", default="result_align")
    parser.add_argument("--out_dir", default="result_report_figs")
    parser.add_argument("--case_lang", choices=["zh", "en"], default="zh")
    parser.add_argument("--case_n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_plot_style()
    configure_chinese_font()

    ensure_dir(args.out_dir)
    fig1_training_curves(
        args.result_zh,
        args.result_en,
        os.path.join(args.out_dir, "fig1_training_curves.png"),
    )
    fig2_model_overlap(
        args.result_zh,
        args.result_en,
        os.path.join(args.out_dir, "fig2_model_overlap.png"),
    )
    fig3_alignment_quality(
        args.result_align,
        os.path.join(args.out_dir, "fig3_alignment_quality.png"),
    )
    case_result_dir = args.result_zh if args.case_lang == "zh" else args.result_en
    fig4_case_table(
        case_result_dir,
        os.path.join(args.out_dir, "fig4_case_table.png"),
        sample_n=args.case_n,
        seed=args.seed,
    )
    print(f"saved 4 figures to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
