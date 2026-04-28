import json
import os
import re
from collections import Counter
from typing import List, Tuple

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
BRACKET_TAG_RE = re.compile(r"\][a-z]+")


def _extract_words_from_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []

    # Remove phrase closing tags like "]nt" and drop "[" markers.
    line = BRACKET_TAG_RE.sub("", line)
    line = line.replace("[", " ")

    parts = re.split(r"\s+", line)
    words = []
    for part in parts:
        if not part:
            continue
        if "/" in part:
            word = part.split("/", 1)[0]
        else:
            word = part
        if word and CJK_RE.search(word):
            words.append(word)
    return words


def load_tokens(pfr_path: str) -> List[str]:
    tokens: List[str] = []
    with open(pfr_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens.extend(_extract_words_from_line(line))
    return tokens


def save_tokens(tokens: List[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")


def load_tokens_from_file(tokens_path: str) -> List[str]:
    tokens: List[str] = []
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    return tokens


def build_vocab(tokens: List[str], vocab_size: int) -> Tuple[List[str], dict]:
    counter = Counter(tokens)
    most_common = [w for w, _ in counter.most_common(max(vocab_size - 2, 0))]
    itos = ["<PAD>", "<UNK>"] + most_common
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def save_vocab(itos: List[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"itos": itos}, f, ensure_ascii=False, indent=2)


def load_vocab(vocab_path: str) -> Tuple[List[str], dict]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    itos = data["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode_tokens(tokens: List[str], stoi: dict) -> List[int]:
    unk = stoi.get("<UNK>", 1)
    return [stoi.get(t, unk) for t in tokens]
