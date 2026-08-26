"""
Twitter Sentiment Analyzer — GUI built on top of the from-scratch Keras
Transformer trained in sentiment_model_ipynb.

Why this file rebuilds the architecture instead of calling
tf.keras.models.load_model() directly:
The custom layers (PositionalEmbedding, TransformerEncoder) used in the
notebook don't override get_config(), so a generic load_model() call can
fail to reconstruct them. Instead we rebuild the exact same architecture
in code and load only the trained weights on top of it — which is robust
regardless of how the custom layers were (or weren't) serialized.

This also means the TextVectorization layer's vocabulary — which was
never saved separately in the notebook — has to be rebuilt here. We do
that by re-running the *exact* same preprocessing + split used at
training time (same CSV, same train_test_split params, same adapt()
call), so the resulting vocabulary — and therefore the Embedding layer's
shape — lines up with the saved weights.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Paths & training-time hyperparameters (must match the notebook exactly)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_scratch_sentiment.keras"
TRAIN_CSV_PATH = BASE_DIR / "data" / "twitter_training_clean_binary.csv"

SEQ_LEN, EMBED_DIM, NUM_HEADS, FF_DIM = 40, 128, 4, 128
MAX_TOKENS = 20000
CLASS_NAMES = ["Negative", "Positive"]

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom layers — identical to the notebook definitions
# ----------------------------------------------------------------------
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[-1])
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.norm1(x + self.drop1(self.att(x, x), training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))

    def get_config(self):
        config = super().get_config()
        config.update(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout_rate,
        )
        return config


def build_scratch_transformer(vocab_size):
    inputs = tf.keras.Input(shape=(SEQ_LEN,))
    x = PositionalEmbedding(SEQ_LEN, vocab_size, EMBED_DIM)(inputs)
    x = TransformerEncoder(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="scratch_transformer")


# ----------------------------------------------------------------------
# Load & cache the vectorizer + model (heavy, so cached across reruns)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="در حال بارگذاری مدل و واژگان...")
def load_artifacts():
    df = pd.read_csv(TRAIN_CSV_PATH)
    df = df[df["sentiment"].isin(CLASS_NAMES)].reset_index(drop=True)

    train_df, _val_df = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=42
    )

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN
    )
    vectorizer.adapt(train_df["text"].astype(str).values)
    vocab_size = len(vectorizer.get_vocabulary())

    model = build_scratch_transformer(vocab_size)
    model.load_weights(str(MODEL_PATH))

    return vectorizer, model, vocab_size, len(train_df)


def predict(texts, vectorizer, model):
    seqs = vectorizer(tf.constant(texts))
    probs = model.predict(seqs, verbose=0).ravel()
    return probs


# ----------------------------------------------------------------------
# Styling — dark theme + signature watermark
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .signature-box {
        position: fixed;
        top: 12px;
        right: 18px;
        z-index: 999;
        text-align: right;
        line-height: 1.1;
        pointer-events: none;
        opacity: 0.92;
    }
    .signature-box .tag {
        font-size: 11px;
        letter-spacing: 3px;
        font-weight: 700;
        color: #9aa0a6;
        text-transform: uppercase;
    }
    .signature-box .name {
        font-family: 'Permanent Marker', cursive;
        font-size: 20px;
        color: #f1f1f1;
        margin-top: 2px;
    }

    .result-card {
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-positive { background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.03)); border-color: rgba(34,197,94,0.35); }
    .result-negative { background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.03)); border-color: rgba(239,68,68,0.35); }

    .result-label { font-size: 26px; font-weight: 700; }
    .result-sub { color: #9aa0a6; font-size: 14px; margin-top: 4px; }
    </style>

    <div class="signature-box">
        <div class="tag">Officiall</div>
        <div class="name">Mahan Liaghatmand</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load model (once)
# ----------------------------------------------------------------------
try:
    vectorizer, model, vocab_size, n_train = load_artifacts()
    load_error = None
except Exception as e:  # surfaced in the UI below instead of crashing silently
    vectorizer = model = vocab_size = n_train = None
    load_error = str(e)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 درباره مدل")
    st.markdown(
        """
        یک **Transformer** ساخته‌شده از پایه با Keras برای تحلیل احساس
        توییت‌ها (مثبت / منفی).
        """
    )
    st.divider()
    st.markdown("**معماری**")
    st.code(
        f"""Seq Len:      {SEQ_LEN}
Embed Dim:    {EMBED_DIM}
Num Heads:    {NUM_HEADS}
FFN Dim:      {FF_DIM}
Vocab Size:   {vocab_size if vocab_size else "—"}
Train Samples:{n_train if n_train else "—"}""",
        language="text",
    )
    st.divider()
    st.markdown("**دیتاست**")
    st.caption("Twitter Sentiment (Positive / Negative) — پیش‌پردازش‌شده")
    st.divider()
    st.caption("Software by Officiall · Mahan Liaghatmand")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("💬 تحلیل احساس توییت")
st.caption("متن یک توییت را وارد کن تا مدل مشخص کند مثبت است یا منفی.")

if load_error:
    st.error(
        "بارگذاری مدل با خطا مواجه شد. مطمئن شو فایل‌های `model/model_scratch_sentiment.keras` "
        "و `data/twitter_training_clean_binary.csv` کنار app.py قرار دارند و tensorflow نصب است.\n\n"
        f"جزئیات خطا:\n```\n{load_error}\n```"
    )
    st.stop()

tab_single, tab_batch = st.tabs(["🔎 تحلیل تکی", "📄 تحلیل دسته‌ای (CSV)"])

with tab_single:
    examples = [
        "I absolutely love this game, best purchase ever!",
        "This update ruined everything, so disappointed.",
        "im getting on borderlands and i will murder you all",
    ]

    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(f"نمونه {i + 1}", use_container_width=True, key=f"ex_{i}"):
            st.session_state["tweet_input"] = ex

    text = st.text_area(
        "متن توییت",
        key="tweet_input",
        height=120,
        placeholder="مثلاً: I can't stop playing this game, it's amazing!",
    )

    analyze = st.button("🚀 تحلیل کن", type="primary", use_container_width=True)

    if analyze:
        clean = (text or "").strip()
        if not clean:
            st.warning("یک متن وارد کن.")
        else:
            prob = float(predict([clean], vectorizer, model)[0])
            label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
            confidence = prob if prob > 0.5 else 1 - prob
            css_class = "result-positive" if label == "Positive" else "result-negative"
            emoji = "😊" if label == "Positive" else "😠"

            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div class="result-label">{emoji} {label}</div>
                    <div class="result-sub">اطمینان مدل: {confidence * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.caption(f"خروجی خام مدل (احتمال کلاس Positive): {prob:.4f}")

with tab_batch:
    st.markdown("یک فایل CSV با ستون `text` آپلود کن تا برای همه ردیف‌ها احساس پیش‌بینی شود.")
    up = st.file_uploader("فایل CSV", type=["csv"])
    if up is not None:
        try:
            batch_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"خواندن فایل ناموفق بود: {e}")
            batch_df = None

        if batch_df is not None:
            if "text" not in batch_df.columns:
                st.error("فایل باید ستونی به نام `text` داشته باشد.")
            else:
                if st.button("🚀 تحلیل دسته‌ای", type="primary"):
                    texts = batch_df["text"].astype(str).tolist()
                    with st.spinner(f"در حال تحلیل {len(texts)} ردیف..."):
                        probs = predict(texts, vectorizer, model)
                    batch_df["sentiment"] = np.where(probs > 0.5, "Positive", "Negative")
                    batch_df["confidence"] = np.where(probs > 0.5, probs, 1 - probs).round(4)
                    st.success("انجام شد.")
                    st.dataframe(batch_df, use_container_width=True)
                    st.download_button(
                        "⬇️ دانلود نتایج (CSV)",
                        batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
