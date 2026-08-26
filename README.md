# 💬 Twitter Sentiment Analysis — Transformer from Scratch vs. Pretrained LLMs

A binary sentiment classifier (Positive / Negative) for tweets, built around a
**Transformer encoder implemented from scratch in Keras** and benchmarked
against off-the-shelf pretrained language models (**BERT**, **DistilBERT**).
Ships with a **Streamlit GUI** for interactive and batch inference.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-HuggingFace-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## Overview

This project trains a compact Transformer encoder **from the ground up**
(no pretrained weights, no Hugging Face backbone) to classify tweet sentiment,
then compares it against two general-purpose pretrained transformer models
used out of the box. The goal was to understand what a Transformer actually
does internally — positional embeddings, multi-head self-attention,
layer normalization, feed-forward blocks — by building every piece by hand,
and to see how a small model trained specifically for this task stacks up
against much larger pretrained models that were never fine-tuned on it.

## Results

Evaluated on a held-out test set of 542 tweets:

| Model                  | Accuracy | Precision | Recall | Notes |
|-------------------------|:--------:|:---------:|:------:|-------|
| **Scratch Transformer** | **93.7%** | **95.2%** | 92.4%  | Trained from scratch on this dataset |
| BERT-Base (uncased)     | 50.0%    | 50.5%     | 92.8%  | Pretrained backbone, **not fine-tuned** — random classification head |
| DistilBERT-Base         | 53.3%    | 52.4%     | 92.8%  | Pretrained backbone, **not fine-tuned** — random classification head |

**Takeaway:** a small task-specific Transformer trained end-to-end on
~36K labeled tweets clearly outperforms large pretrained models used as
zero-shot classifiers, because BERT/DistilBERT were loaded with a fresh,
untrained classification head rather than fine-tuned on this data. This
comparison is best read as *"purpose-built small model vs. pretrained models
with zero task adaptation,"* not as a verdict on pretrained models in
general — fine-tuning BERT/DistilBERT on the same data would very likely
close or reverse this gap. That's a natural next step (see
[Future Work](#future-work)).

## Architecture

The scratch model is a single-block Transformer encoder for binary
sequence classification:

```
Input (tokenized, seq_len=40)
   │
   ▼
TextVectorization (vocab, max_tokens=20,000)
   │
   ▼
Positional Embedding  (token embedding + learned positional embedding)
   │
   ▼
Transformer Encoder Block
   ├─ Multi-Head Self-Attention (4 heads)
   ├─ Add & LayerNorm
   ├─ Feed-Forward Network (Dense 128 → Dense 128)
   └─ Add & LayerNorm
   │
   ▼
Global Average Pooling
   │
   ▼
Dropout (0.3) → Dense(64, relu) → Dense(1, sigmoid)
   │
   ▼
Positive / Negative
```

| Hyperparameter      | Value |
|----------------------|-------|
| Sequence length       | 40 |
| Embedding dimension    | 128 |
| Attention heads        | 4 |
| Feed-forward dimension | 128 |
| Optimizer              | Adam |
| Loss                   | Binary cross-entropy |
| Callbacks              | EarlyStopping, ReduceLROnPlateau |

## Dataset

- Source: Twitter sentiment dataset (tweet text + sentiment label).
- Original labels: `Positive`, `Neutral`, `Negative`, `Irrelevant` — this
  project keeps only `Positive` / `Negative` for a binary classification task.
- Split: 90% train / 10% validation (stratified, `random_state=42`), plus a
  separate held-out test set.
- Final size: ~36K training tweets, ~4K validation, 542 test.

## Project Structure

```
.
├── notebooks/
│   └── sentiment_model.ipynb        # data prep, training, evaluation
├── app/
│   ├── app.py                       # Streamlit GUI (single + batch inference)
│   ├── requirements.txt
│   ├── model/
│   │   └── model_scratch_sentiment.keras
│   └── data/
│       └── twitter_training_clean_binary.csv   # needed to rebuild the vectorizer
└── README.md
```

## Getting Started

### 1. Train / reproduce the notebook

```bash
pip install tensorflow torch transformers scikit-learn pandas matplotlib
jupyter notebook notebooks/sentiment_model.ipynb
```

### 2. Run the GUI

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app rebuilds the `TextVectorization` vocabulary from the training CSV at
startup (using the exact same split and settings as training) and loads the
saved model weights on top of a freshly-built architecture — see the note in
`app/README.md` for why.

**Features:**
- Single-tweet analysis with a confidence bar
- Batch analysis via CSV upload (`text` column in, `sentiment` +
  `confidence` columns out, downloadable)
- Sidebar with live architecture/vocab stats

## Tech Stack

- **TensorFlow / Keras** — custom Transformer encoder, training pipeline
- **PyTorch + Hugging Face Transformers** — pretrained BERT / DistilBERT baselines
- **scikit-learn** — train/test split, evaluation metrics
- **Streamlit** — interactive GUI

## Future Work

- Fine-tune BERT/DistilBERT on this dataset for an apples-to-apples comparison
- Extend back to the original 4-class problem (Positive / Neutral / Negative / Irrelevant)
- Swap `TextVectorization` for a subword tokenizer (WordPiece/BPE) to reduce OOV
- Persist the vectorizer directly (e.g. as a standalone `.keras` layer or pickled vocab) instead of rebuilding it from the training split at load time
- Add ONNX/TFLite export for lighter-weight deployment

## License

MIT — feel free to use, modify, and learn from this project.

---

<sub>Software by **Officiall** · Mahan Liaghatmand</sub>
