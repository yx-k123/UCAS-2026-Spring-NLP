# Word Vector Training (Chinese PFR)

## 1) Prepare tokens and vocab

```bash
cd pj2
python prepare_pfr.py --input data/raw/PeopleDaily199801.txt --tokens_out data/processed/pfr_tokens.txt --vocab_out data/processed/pfr_vocab.json --vocab_size 1000
```

Or use config files:

```bash
python prepare_pfr.py --config configs/prepare.json
```

For English JSON corpus (e.g. from `pj1/data/bbc_data_2000.json`):

```bash
python prepare_pfr.py --lang en --input ../pj1/data/bbc_data_2000.json --json_field content --tokens_out data/processed/en_tokens.txt --vocab_out data/processed/en_vocab.json --vocab_size 10000
```

## 2) Train models

```bash
python train_fnn.py --tokens data/processed/pfr_tokens.txt --vocab_path data/processed/pfr_vocab.json --embed_dim 10
python train_rnn.py --tokens data/processed/pfr_tokens.txt --vocab_path data/processed/pfr_vocab.json --embed_dim 10
python train_lstm.py --tokens data/processed/pfr_tokens.txt --vocab_path data/processed/pfr_vocab.json --embed_dim 10
```

All training scripts now support shared options:
- `--seed`
- `--dropout`
- `--grad_clip`
- `--run_name`

Each run writes unified logs to `result/train_<model>[_run_name].jsonl`.

Or use config files:

```bash
python train_fnn.py --config configs/fnn.json
python train_rnn.py --config configs/rnn.json
python train_lstm.py --config configs/lstm.json
```

## 3) Inspect similar words

```bash
python eval_vectors.py --model fnn
python eval_vectors.py --model rnn
python eval_vectors.py --model lstm
```

Outputs are saved under the result/ directory.

## 4) Cross-lingual alignment (optional)

```bash
python align_bilingual.py \
  --src_vocab data/processed/pfr_vocab.json \
  --src_emb result/lstm_embeddings.pt \
  --tgt_vocab data/processed/en_vocab.json \
  --tgt_emb result_en/lstm_embeddings.pt \
  --lexicon configs/zh_en_seed_lexicon.txt \
  --report_out result/zh_en_alignment.txt
```

## Layout

```
pj2/
	data/
		raw/        # original corpus files
		processed/  # tokens and vocab
	result/       # embeddings, similar words, tensorboard logs
	configs/      # json config files
```
