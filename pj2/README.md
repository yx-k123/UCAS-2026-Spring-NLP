# pj2：词向量训练与对比实验说明（中文）

本项目用于完成以下任务：
- 基于中文或英文语料训练词向量；
- 对比 `FNN(CBOW)`、`RNN LM`、`LSTM LM` 三种模型的词向量差异；
- 对随机词汇输出 Top-K 相似词并做人工分析；
- 可选：进行中英词向量空间对齐，比较同义词（如“书”与 `book`）的距离。

## 1. 目录结构

```text
pj2/
├── configs/                  # 训练与预处理配置
├── data/
│   ├── raw/                  # 原始语料（如人民日报）
│   └── processed/            # 处理后的 tokens/vocab
├── result/                   # 中文训练结果
├── result_en/                # 英文训练结果（建议）
├── prepare_pfr.py            # 语料预处理（支持中/英）
├── train_fnn.py              # FNN(CBOW) 训练
├── train_rnn.py              # RNN 语言模型训练
├── train_lstm.py             # LSTM 语言模型训练
├── eval_vectors.py           # 相似词评估
├── align_bilingual.py        # 中英对齐（Procrustes）
└── train_utils.py            # seed、统一日志等工具
```

## 2. 环境与运行位置

建议在项目根目录执行：

```bash
cd /home/kouyx/NLP/pj2
```

依赖以 `torch` 为主。如果你的环境中尚未安装，请先安装对应版本的 PyTorch。

当前提供的核心配置文件：
- `configs/prepare_zh.json`、`configs/prepare_en.json`
- `configs/fnn_zh.json`、`configs/rnn_zh.json`、`configs/lstm_zh.json`
- `configs/fnn_en.json`、`configs/rnn_en.json`、`configs/lstm_en.json`
- `configs/align_zh_en.json`

## 3. 语料预处理（推荐使用 config）

### 3.1 中文语料（人民日报）

```bash
python prepare_pfr.py --config configs/prepare_zh.json
```

### 3.2 英文语料（JSON，如 pj1 的 bbc 数据）

```bash
python prepare_pfr.py --config configs/prepare_en.json
```

说明：
- `--lang en` 时会从 JSON 的 `content` 字段抽取 `[a-z]+` 单词并转小写；
- 产出格式与中文一致：`tokens.txt`（每行一个 token）和 `vocab.json`。

## 4. 模型训练

### 4.1 中文训练（config）

```bash
python train_fnn.py --config configs/fnn_zh.json
python train_rnn.py --config configs/rnn_zh.json
python train_lstm.py --config configs/lstm_zh.json
```

### 4.2 英文训练（config）

```bash
python train_fnn.py --config configs/fnn_en.json
python train_rnn.py --config configs/rnn_en.json
python train_lstm.py --config configs/lstm_en.json
```

### 4.3 统一训练参数说明

三个训练脚本都支持：
- `--seed`：固定随机种子，保证可复现；
- `--dropout`：dropout 比例；
- `--grad_clip`：梯度裁剪阈值（建议 `1.0`）；
- `--run_name`：运行名，用于区分日志文件。

统一日志输出：
- `result/train_<model>[_run_name].jsonl`（每个 epoch 一行）
- `result/train_<model>[_run_name]_summary.json`（配置与输出路径汇总）

## 5. 相似词评估

中文：

```bash
python eval_vectors.py --model fnn  --result_dir result_zh --vocab_path data/processed/pfr_vocab.json
python eval_vectors.py --model rnn  --result_dir result_zh --vocab_path data/processed/pfr_vocab.json
python eval_vectors.py --model lstm --result_dir result_zh --vocab_path data/processed/pfr_vocab.json
```

英文：

```bash
python eval_vectors.py --model fnn  --result_dir result_en --vocab_path data/processed/en_vocab.json
python eval_vectors.py --model rnn  --result_dir result_en --vocab_path data/processed/en_vocab.json
python eval_vectors.py --model lstm --result_dir result_en --vocab_path data/processed/en_vocab.json
```

输出文件：
- `similar_fnn.txt`
- `similar_rnn.txt`
- `similar_lstm.txt`

每行格式：`目标词\t近邻词1,近邻词2,...`

## 6. 中英对齐与同义词距离比较（可选）

先准备种子词典（已提供模板）：
- `configs/zh_en_seed_lexicon.txt`（格式：`中文词<TAB>英文词`）

运行对齐（config）：

```bash
python align_bilingual.py --config configs/align_zh_en.json
```

输出内容包括：
- 锚点词对的 `cosine` 和 `l2` 距离；
- 对齐后每个中文锚点词在英文空间的 Top-K 邻居。

## 7. 实验报告建议写法

建议按以下结构整理：
1. 数据来源与规模（中文、英文 token 数与词表大小）。  
2. 模型设置（FNN/RNN/LSTM 参数统一和差异）。  
3. 同一词集合的近邻对比（可展示 20 个词）。  
4. 人工判断一致性分析（语义相近、词性偏差、高频词干扰）。  
5. 中英对齐后同义词距离分析（如“书-book”“工作-work/job”）。

## 8. 常见问题

1. 三种模型相似词差异很大是否正常？  
正常。因为 FNN 与 RNN/LSTM 的训练目标不同，向量空间不完全一致是预期现象。

2. 为什么有些词近邻看起来不合理？  
通常是低频词、语料噪声、词表过小（`<UNK>` 过多）或训练轮数不足导致。

3. 能否直接比较“书”和 `book` 的距离？  
不能直接比。必须先做跨语言对齐（如 `align_bilingual.py`），否则不在同一向量空间。
