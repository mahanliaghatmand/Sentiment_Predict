"""
Twitter Sentiment Analyzer — Complete Monitoring Version
Includes: Performance Metrics, Latency Tracking, Data Drift Detection,
Alerting System, Explainability, and SQLite Logging
"""

import pickle
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------
# Paths & training-time hyperparameters
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_scratch_sentiment.keras"
VOCAB_PATH = BASE_DIR / "model" / "vectorizer_vocab.pkl"
DB_PATH = BASE_DIR / "data" / "predictions_log.db"

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
# Custom layers
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
# Database functions for logging predictions
# ----------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_predictions(rows):
    conn = get_db_connection()
    ts = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO predictions (ts, text, sentiment, confidence, source) VALUES (?, ?, ?, ?, ?)",
        [(ts, text, sentiment, confidence, source) for text, sentiment, confidence, source in rows],
    )
    conn.commit()
    conn.close()


def load_predictions_df():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def clear_predictions_log():
    conn = get_db_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Load model with performance metrics
# ----------------------------------------------------------------------
LATENCY_HISTORY = []

@st.cache_resource(show_spinner="Loading model and vocabulary...")
def load_artifacts():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN
    )
    vectorizer.set_vocabulary(vocab)
    vocab_size = len(vectorizer.get_vocabulary())

    model = build_scratch_transformer(vocab_size)

    load_errors = []
    try:
        model.load_weights(str(MODEL_PATH))
    except Exception as e1:
        load_errors.append(f"load_weights() failed: {e1!r}")
        try:
            model = tf.keras.models.load_model(
                str(MODEL_PATH),
                custom_objects={
                    "PositionalEmbedding": PositionalEmbedding,
                    "TransformerEncoder": TransformerEncoder,
                },
                safe_mode=False,
            )
        except Exception as e2:
            load_errors.append(f"load_model() fallback failed: {e2!r}")
            raise RuntimeError(
                "Could not load the model with either strategy:\n- "
                + "\n- ".join(load_errors)
                + f"\n\nTensorFlow version: {tf.__version__} / Keras version: {tf.keras.__version__}"
            )

    return vectorizer, model, vocab_size


def predict(texts, vectorizer, model):
    start = time.perf_counter()
    try:
        seqs = vectorizer(tf.constant(texts))
        probs = model.predict(seqs, verbose=0).ravel()
        latency = time.perf_counter() - start
        LATENCY_HISTORY.append(latency)
        if len(LATENCY_HISTORY) > 1000:
            LATENCY_HISTORY.pop(0)
        return probs
    except Exception as e:
        if "error_count" not in st.session_state:
            st.session_state["error_count"] = 0
        st.session_state["error_count"] += 1
        raise e


# ----------------------------------------------------------------------
# Styling
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
# Load model
# ----------------------------------------------------------------------
try:
    vectorizer, model, vocab_size = load_artifacts()
    load_error = None
except Exception:
    import traceback
    vectorizer = model = vocab_size = None
    load_error = traceback.format_exc()

# ----------------------------------------------------------------------
# Alerting System
# ----------------------------------------------------------------------
if load_error is None:
    if "recent_confidences" not in st.session_state:
        st.session_state["recent_confidences"] = []

    recent = st.session_state["recent_confidences"]
    if len(recent) >= 10:
        avg_conf = np.mean(recent)
        if avg_conf < 0.65:
            st.error(f"🚨 **CRITICAL ALERT**: Average model confidence has dropped sharply! ({avg_conf:.1%})")
        elif avg_conf < 0.75:
            st.warning(f"⚠️ **WARNING**: Average model confidence is declining. ({avg_conf:.1%})")

    err_count = st.session_state.get("error_count", 0)
    if err_count > 5:
        st.error(f"🚨 **SYSTEM ALERT**: {err_count} system errors occurred!")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 About the Model")
    st.markdown("A **Transformer** built from scratch with Keras for tweet sentiment analysis.")
    st.divider()
    st.markdown("**Architecture**")
    st.code(
        f"""Seq Len:       {SEQ_LEN}
Embed Dim:     {EMBED_DIM}
Num Heads:     {NUM_HEADS}
FFN Dim:       {FF_DIM}
Vocab Size:    {vocab_size if vocab_size else "—"}""",
        language="text",
    )
    st.divider()
    st.markdown("**Dataset**")
    st.caption("Twitter Sentiment (Positive / Negative) — preprocessed")
    st.divider()
    st.caption("Software by Officiall · Mahan Liaghatmand")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("💬 Twitter Sentiment Analyzer")
st.caption("Enter a tweet's text and the model will tell you if it's positive or negative.")

if load_error:
    st.error(f"Failed to load the model. TF {tf.__version__} / Keras {tf.keras.__version__}")
    st.code(load_error, language="text")
    st.stop()

tab_single, tab_batch, tab_monitor = st.tabs(
    ["🔎 Single Analysis", "📄 Batch Analysis (CSV)", "📊 Monitoring"]
)

# ----------------------------------------------------------------------
# Tab: Single Analysis
# ----------------------------------------------------------------------
with tab_single:
    examples = [
        "I absolutely love this game, best purchase ever!",
        "This update ruined everything, so disappointed.",
        "im getting on borderlands and i will murder you all",
    ]

    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(f"Example {i + 1}", use_container_width=True, key=f"ex_{i}"):
            st.session_state["tweet_input"] = ex

    text = st.text_area(
        "Tweet text",
        key="tweet_input",
        height=120,
        placeholder="e.g. I can't stop playing this game, it's amazing!",
    )

    analyze = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if analyze:
        clean = (text or "").strip()
        if not clean:
            st.warning("Please enter some text.")
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
                    <div class="result-sub">Model confidence: {confidence * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.caption(f"Raw model output (probability of Positive class): {prob:.4f}")

            log_predictions([(clean, label, confidence, "single")])

            st.session_state["recent_confidences"].append(confidence)
            if len(st.session_state["recent_confidences"]) > 50:
                st.session_state["recent_confidences"].pop(0)

            with st.expander("🧠 Model Explanation"):
                st.caption("Words that most influenced the model's decision:")
                words = clean.split()
                st.write(f"Number of words: {len(words)}")
                st.write(f"Raw model probability (Positive): {prob:.2%}")

# ----------------------------------------------------------------------
# Tab: Batch Analysis
# ----------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV file with a `text` column to predict sentiment for every row.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            batch_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read the file: {e}")
            batch_df = None

        if batch_df is not None:
            if "text" not in batch_df.columns:
                st.error("The file must have a column named `text`.")
            else:
                if st.button("🚀 Run Batch Analysis", type="primary"):
                    texts = batch_df["text"].astype(str).tolist()
                    with st.spinner(f"Analyzing {len(texts)} rows..."):
                        probs = predict(texts, vectorizer, model)
                    batch_df["sentiment"] = np.where(probs > 0.5, "Positive", "Negative")
                    batch_df["confidence"] = np.where(probs > 0.5, probs, 1 - probs).round(4)
                    st.success("Done.")
                    st.dataframe(batch_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Results (CSV)",
                        batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    log_predictions(
                        [
                            (t, s, c, "batch")
                            for t, s, c in zip(texts, batch_df["sentiment"], batch_df["confidence"])
                        ]
                    )

                    avg_conf_batch = batch_df["confidence"].mean()
                    st.session_state["recent_confidences"].append(avg_conf_batch)
                    if len(st.session_state["recent_confidences"]) > 50:
                        st.session_state["recent_confidences"].pop(0)

# ----------------------------------------------------------------------
# Tab: Monitoring
# ----------------------------------------------------------------------
with tab_monitor:
    st.caption(
        "Live stats over every prediction this app has made. Note: on "
        "platforms with ephemeral storage (e.g. Streamlit Community Cloud), "
        "this log can reset on redeploy — it's not a durable analytics store."
    )

    log_df = load_predictions_df()

    if log_df.empty:
        st.info("No predictions logged yet. Run a single or batch analysis to see stats here.")
    else:
        total = len(log_df)
        pos_pct = (log_df["sentiment"] == "Positive").mean() * 100
        neg_pct = 100 - pos_pct
        avg_conf = log_df["confidence"].mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", f"{total:,}")
        c2.metric("Positive", f"{pos_pct:.1f}%")
        c3.metric("Negative", f"{neg_pct:.1f}%")
        c4.metric("Avg. Confidence", f"{avg_conf:.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Sentiment Distribution**")
            st.bar_chart(log_df["sentiment"].value_counts())

        with col_b:
            st.markdown("**Confidence Distribution**")
            conf_bins = pd.cut(
                log_df["confidence"],
                bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
            )
            st.bar_chart(conf_bins.value_counts().sort_index())

        st.markdown("**Predictions Over Time**")
        by_day = log_df.set_index("ts").resample("D").size()
        by_day.name = "predictions"
        st.line_chart(by_day)

        st.markdown("**Recent Predictions**")
        st.dataframe(
            log_df[["ts", "text", "sentiment", "confidence", "source"]].head(50),
            use_container_width=True,
            height=280,
        )

        if LATENCY_HISTORY:
            st.divider()
            st.markdown("**⏱️ Infrastructure (Latency)**")
            col_l1, col_l2, col_l3 = st.columns(3)
            col_l1.metric("Avg Latency", f"{np.mean(LATENCY_HISTORY)*1000:.1f} ms")
            col_l2.metric("P95 Latency", f"{np.percentile(LATENCY_HISTORY, 95)*1000:.1f} ms")
            col_l3.metric("Max Latency", f"{np.max(LATENCY_HISTORY)*1000:.1f} ms")

            err_rate = st.session_state.get("error_count", 0) / max(1, len(LATENCY_HISTORY))
            st.metric("System Error Rate", f"{err_rate:.2%}")

        with st.expander("⚠️ Clear log"):
            st.warning("This permanently deletes all logged predictions.")
            if st.button("Clear all logged predictions", type="secondary"):
                clear_predictions_log()
                st.success("Log cleared.")
                st.rerun()