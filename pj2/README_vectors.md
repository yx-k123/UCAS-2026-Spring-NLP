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

## 2) Train models

```bash
python train_fnn.py --tokens data/processed/pfr_tokens.txt --vocab_path data/processed/pfr_vocab.json --embed_dim 10
python train_rnn.py --tokens data/processed/pfr_tokens.txt --vocab_path data/processed/pfr_vocab.json --embed_dim 10
python train_lstm.py --tokens data/processed/pfr_tokens.txt --vocab_path data/processed/pfr_vocab.json --embed_dim 10
```

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

## Layout

```
pj2/
	data/
		raw/        # original corpus files
		processed/  # tokens and vocab
	result/       # embeddings, similar words, tensorboard logs
	configs/      # json config files
```
