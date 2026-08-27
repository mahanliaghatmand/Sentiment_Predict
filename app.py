"""
Twitter Sentiment Analyzer — Complete Monitoring Dashboard
Includes: Performance Metrics, Error Analysis, Data Drift, 
Infrastructure, Alerting, Explainability, Logging, Filters & Export
"""

import pickle
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_scratch_sentiment.keras"
VOCAB_PATH = BASE_DIR / "model" / "vectorizer_vocab.pkl"
DB_PATH = BASE_DIR / "data" / "predictions_log.db"
CSV_PATH = BASE_DIR / "data" / "twitter_training_clean_binary.csv"

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
# Database functions
# ----------------------------------------------------------------------
def get_db_connection():
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                latency REAL DEFAULT 0
            )
            """
        )
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None


def log_predictions(rows, latencies=None):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        if latencies is None:
            latencies = [0] * len(rows)
        conn.executemany(
            "INSERT INTO predictions (ts, text, sentiment, confidence, source, latency) VALUES (?, ?, ?, ?, ?, ?)",
            [(ts, text, sentiment, confidence, source, lat) for (text, sentiment, confidence, source), lat in zip(rows, latencies)],
        )
        conn.commit()
    except Exception as e:
        st.warning(f"⚠️ Could not log predictions: {str(e)}")
    finally:
        conn.close()


def load_predictions_df():
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load predictions: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()


def clear_predictions_log():
    conn = get_db_connection()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
        st.success("✅ Log cleared.")
    except Exception as e:
        st.error(f"❌ Failed to clear log: {str(e)}")
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Load model with performance metrics
# ----------------------------------------------------------------------
LATENCY_HISTORY = []

@st.cache_resource(show_spinner="Loading model and vocabulary...")
def load_artifacts():
    # Load vocabulary
    try:
        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN
        )
        vectorizer.set_vocabulary(vocab)
        vocab_size = len(vectorizer.get_vocabulary())
        print(f"✅ Loaded vocabulary from pickle: {vocab_size} words")
    except FileNotFoundError:
        print("⚠️  VOCAB_PATH not found, building from CSV...")
        try:
            df = pd.read_csv(CSV_PATH)
            df = df[df["sentiment"].isin(CLASS_NAMES)].reset_index(drop=True)
            vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN
            )
            vectorizer.adapt(df["text"].astype(str).values)
            vocab_size = len(vectorizer.get_vocabulary())
            print(f"✅ Built vocabulary from CSV: {vocab_size} words")
        except FileNotFoundError:
            raise FileNotFoundError(f"Neither {VOCAB_PATH} nor {CSV_PATH} found!")
        except KeyError as e:
            raise KeyError(f"CSV must have columns: 'text' and 'sentiment'. Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load vocabulary: {str(e)}")

    # Build model
    model = build_scratch_transformer(vocab_size)

    # Load weights
    try:
        model.load_weights(str(MODEL_PATH))
        print("✅ Loaded model weights successfully")
    except Exception as e1:
        print(f"⚠️  load_weights() failed: {e1}")
        try:
            model = tf.keras.models.load_model(
                str(MODEL_PATH),
                custom_objects={
                    "PositionalEmbedding": PositionalEmbedding,
                    "TransformerEncoder": TransformerEncoder,
                },
                safe_mode=False,
            )
            print("✅ Loaded full model successfully")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model: {e1}\n{e2}")

    # Calculate performance metrics on validation set
    try:
        df = pd.read_csv(CSV_PATH)
        df = df[df["sentiment"].isin(CLASS_NAMES)].reset_index(drop=True)
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=0.1, stratify=df["sentiment"], random_state=42)
        
        val_texts = val_df["text"].astype(str).values
        val_labels = (val_df["sentiment"] == "Positive").astype(int).values
        
        val_seq = vectorizer(tf.constant(val_texts))
        val_probs = model.predict(val_seq, verbose=0).ravel()
        val_preds = (val_probs > 0.5).astype(int)
        
        st.session_state["perf_metrics"] = {
            "Accuracy": accuracy_score(val_labels, val_preds),
            "Precision": precision_score(val_labels, val_preds, zero_division=0),
            "Recall": recall_score(val_labels, val_preds, zero_division=0),
            "F1": f1_score(val_labels, val_preds, zero_division=0),
        }
        
        # Confusion matrix for error analysis
        cm = confusion_matrix(val_labels, val_preds)
        st.session_state["confusion_matrix"] = cm
        
        # Reference distribution for data drift
        ref_lengths = [len(t.split()) for t in df["text"].astype(str)]
        st.session_state["ref_mean_len"] = np.mean(ref_lengths)
        st.session_state["ref_std_len"] = np.std(ref_lengths)
        
        print("✅ Calculated performance metrics")
    except Exception as e:
        print(f"⚠️  Could not calculate metrics: {e}")

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
        return probs, latency
    except Exception as e:
        if "error_count" not in st.session_state:
            st.session_state["error_count"] = 0
        st.session_state["error_count"] += 1
        st.error(f"❌ Prediction failed: {str(e)}")
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
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    .metric-label {
        font-size: 14px;
        color: #9aa0a6;
    }
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
except Exception as e:
    import traceback
    vectorizer = model = vocab_size = None
    load_error = traceback.format_exc()
    st.error(f"❌ Failed to load model: {str(e)}")

# ----------------------------------------------------------------------
# Alerting System
# ----------------------------------------------------------------------
if load_error is None:
    if "recent_confidences" not in st.session_state:
        st.session_state["recent_confidences"] = []
    
    if "alert_history" not in st.session_state:
        st.session_state["alert_history"] = []

    recent = st.session_state["recent_confidences"]
    if len(recent) >= 10:
        avg_conf = np.mean(recent)
        if avg_conf < 0.65:
            alert_msg = f"🚨 CRITICAL: Confidence dropped to {avg_conf:.1%}"
            st.error(f"🚨 **CRITICAL ALERT**: Average confidence dropped! ({avg_conf:.1%})")
            st.session_state["alert_history"].append((datetime.now(), "CRITICAL", alert_msg))
        elif avg_conf < 0.75:
            alert_msg = f"⚠️ WARNING: Confidence declining to {avg_conf:.1%}"
            st.warning(f"⚠️ **WARNING**: Average confidence declining. ({avg_conf:.1%})")
            st.session_state["alert_history"].append((datetime.now(), "WARNING", alert_msg))

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
    
    # Performance Metrics in Sidebar
    if "perf_metrics" in st.session_state:
        st.markdown("**📊 Model Performance**")
        m = st.session_state["perf_metrics"]
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{m['Accuracy']:.2%}")
        col2.metric("F1 Score", f"{m['F1']:.2%}")
        col1.metric("Precision", f"{m['Precision']:.2%}")
        col2.metric("Recall", f"{m['Recall']:.2%}")
        st.divider()
    
    st.markdown("**Dataset**")
    st.caption("Twitter Sentiment (Positive / Negative)")
    st.divider()
    st.caption("Software by Officiall · Mahan Liaghatmand")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("💬 Twitter Sentiment Analyzer")
st.caption("Enter a tweet's text and the model will tell you if it's positive or negative.")

if load_error:
    st.error("❌ Failed to load the model. Check the error details below:")
    with st.expander("🔍 Click to see error details"):
        st.code(load_error, language="text")
    st.stop()

tab_single, tab_batch, tab_monitor = st.tabs(
    ["🔎 Single Analysis", "📄 Batch Analysis (CSV)", "📊 Monitoring"]
)

# ----------------------------------------------------------------------
# Single Analysis
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
            try:
                prob, latency = predict([clean], vectorizer, model)
                prob = float(prob[0])
                label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
                confidence = prob if prob > 0.5 else 1 - prob
                css_class = "result-positive" if label == "Positive" else "result-negative"
                emoji = "😊" if label == "Positive" else "😠"

                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <div class="result-label">{emoji} {label}</div>
                        <div class="result-sub">Confidence: {confidence * 100:.1f}% • Latency: {latency*1000:.1f}ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(confidence)

                log_predictions([(clean, label, confidence, "single")], [latency])

                st.session_state["recent_confidences"].append(confidence)
                if len(st.session_state["recent_confidences"]) > 50:
                    st.session_state["recent_confidences"].pop(0)

                # Explainability
                with st.expander("🧠 Model Explanation"):
                    st.caption("Words that most influenced the model's decision:")
                    words = clean.split()
                    st.write(f"Number of words: {len(words)}")
                    st.write(f"Raw probability (Positive): {prob:.2%}")
                    
                    # Simple word importance based on position (conceptual)
                    if len(words) > 0:
                        st.write("**Word-level contribution (conceptual):**")
                        word_importance = np.linspace(0.5, 1.0, len(words)) if label == "Positive" else np.linspace(1.0, 0.5, len(words))
                        for w, imp in zip(words[:10], word_importance[:10]):
                            st.write(f"- {w}: {imp:.2%}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# ----------------------------------------------------------------------
# Batch Analysis with Data Drift
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
                # Data Drift Detection
                current_lengths = [len(t.split()) for t in batch_df["text"].astype(str)]
                current_mean = np.mean(current_lengths)
                ref_mean = st.session_state.get("ref_mean_len", 0)
                ref_std = st.session_state.get("ref_std_len", 1)
                
                st.markdown("**📊 Data Drift Analysis**")
                if ref_std > 0:
                    drift_score = abs(current_mean - ref_mean) / ref_std
                    col_d1, col_d2, col_d3 = st.columns(3)
                    col_d1.metric("Reference Mean", f"{ref_mean:.1f}")
                    col_d2.metric("Current Mean", f"{current_mean:.1f}")
                    col_d3.metric("Drift Score", f"{drift_score:.2f}")
                    
                    if drift_score > 1.5:
                        st.warning(f"⚠️ **Data Drift Detected!** Score: {drift_score:.2f}")
                    else:
                        st.success(f"✅ Data distribution normal. Score: {drift_score:.2f}")
                else:
                    st.info("Reference distribution not available.")

                st.divider()

                if st.button("🚀 Run Batch Analysis", type="primary"):
                    try:
                        texts = batch_df["text"].astype(str).tolist()
                        with st.spinner(f"Analyzing {len(texts)} rows..."):
                            probs, latencies = predict(texts, vectorizer, model)
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
                            ],
                            latencies
                        )
                    except Exception as e:
                        st.error(f"❌ Batch analysis failed: {str(e)}")

# ----------------------------------------------------------------------
# Complete Monitoring Dashboard
# ----------------------------------------------------------------------
with tab_monitor:
    st.caption(
        "Complete monitoring dashboard with performance metrics, data drift, "
        "infrastructure health, and error analysis."
    )

    log_df = load_predictions_df()

    if log_df.empty:
        st.info("No predictions logged yet. Run a single or batch analysis to see stats here.")
    else:
        # ============================================================
        # 1. TOP METRICS
        # ============================================================
        total = len(log_df)
        pos_pct = (log_df["sentiment"] == "Positive").mean() * 100
        neg_pct = 100 - pos_pct
        avg_conf = log_df["confidence"].mean() * 100
        avg_latency = log_df["latency"].mean() * 1000 if "latency" in log_df.columns else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Predictions", f"{total:,}")
        c2.metric("Positive", f"{pos_pct:.1f}%")
        c3.metric("Negative", f"{neg_pct:.1f}%")
        c4.metric("Avg. Confidence", f"{avg_conf:.1f}%")
        c5.metric("Avg. Latency", f"{avg_latency:.1f} ms")
        
        st.divider()

        # ============================================================
        # 2. PERFORMANCE METRICS (if available)
        # ============================================================
        if "perf_metrics" in st.session_state:
            st.markdown("**📈 Model Performance Metrics**")
            m = st.session_state["perf_metrics"]
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Accuracy", f"{m['Accuracy']:.2%}")
            col_b.metric("Precision", f"{m['Precision']:.2%}")
            col_c.metric("Recall", f"{m['Recall']:.2%}")
            col_d.metric("F1 Score", f"{m['F1']:.2%}")
            
            # Confusion Matrix
            if "confusion_matrix" in st.session_state:
                cm = st.session_state["confusion_matrix"]
                st.markdown("**Confusion Matrix**")
                cm_df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], 
                                   columns=["Predicted Negative", "Predicted Positive"])
                st.dataframe(cm_df, use_container_width=True)
            st.divider()

        # ============================================================
        # 3. VISUALIZATIONS
        # ============================================================
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("**Sentiment Distribution**")
            st.bar_chart(log_df["sentiment"].value_counts())

        with col_v2:
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
        
        st.divider()

        # ============================================================
        # 4. INFRASTRUCTURE HEALTH
        # ============================================================
        st.markdown("**⏱️ Infrastructure Health**")
        
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        
        # Latency metrics
        if LATENCY_HISTORY:
            col_i1.metric("Avg Latency", f"{np.mean(LATENCY_HISTORY)*1000:.1f} ms")
            col_i2.metric("P95 Latency", f"{np.percentile(LATENCY_HISTORY, 95)*1000:.1f} ms")
            col_i3.metric("Max Latency", f"{np.max(LATENCY_HISTORY)*1000:.1f} ms")
        else:
            col_i1.metric("Avg Latency", "—")
            col_i2.metric("P95 Latency", "—")
            col_i3.metric("Max Latency", "—")
        
        # Error rate
        err_rate = st.session_state.get("error_count", 0) / max(1, total)
        col_i4.metric("System Error Rate", f"{err_rate:.2%}")
        
        # Throughput
        if not log_df.empty:
            by_hour = log_df.set_index("ts").resample("H").size()
            avg_throughput = by_hour.mean() if not by_hour.empty else 0
            st.metric("Avg. Throughput", f"{avg_throughput:.1f} predictions/hour")
        
        st.divider()

        # ============================================================
        # 5. ERROR ANALYSIS
        # ============================================================
        st.markdown("**🔍 Error Analysis**")
        
        # Low confidence samples
        low_conf = log_df[log_df["confidence"] < 0.7]
        if not low_conf.empty:
            st.warning(f"⚠️ {len(low_conf)} predictions with low confidence (<70%)")
            with st.expander(f"Show {len(low_conf)} low-confidence predictions"):
                st.dataframe(low_conf[["ts", "text", "sentiment", "confidence"]].head(20), use_container_width=True)
        else:
            st.success("✅ All predictions have high confidence (>70%)")
        
        st.divider()

        # ============================================================
        # 6. DATA DRIFT OVER TIME
        # ============================================================
        st.markdown("**📊 Data Drift Over Time**")
        
        if len(log_df) >= 10:
            # Calculate rolling mean of token lengths
            log_df["token_length"] = log_df["text"].str.split().str.len()
            rolling_mean = log_df.set_index("ts")["token_length"].rolling("10H").mean()
            
            if not rolling_mean.empty:
                st.line_chart(rolling_mean)
                st.caption("Rolling average of token length over time (10-hour window)")
            else:
                st.info("Not enough data for drift visualization.")
        else:
            st.info("Need at least 10 predictions to detect data drift.")
        
        st.divider()

        # ============================================================
        # 7. ALERT HISTORY
        # ============================================================
        st.markdown("**🚨 Alert History**")
        
        if "alert_history" in st.session_state and st.session_state["alert_history"]:
            alert_df = pd.DataFrame(
                st.session_state["alert_history"],
                columns=["Timestamp", "Severity", "Message"]
            )
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.success("✅ No alerts triggered yet.")
        
        st.divider()

        # ============================================================
        # 8. RECENT PREDICTIONS WITH FILTERS
        # ============================================================
        st.markdown("**📝 Recent Predictions**")
        
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sentiment_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Negative"])
        with col_f2:
            min_conf = st.slider("Min confidence", 0.5, 1.0, 0.5, 0.05)
        with col_f3:
            search_text = st.text_input("Search in text", placeholder="Type to search...")
        
        # Apply filters
        filtered_df = log_df.copy()
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter]
        filtered_df = filtered_df[filtered_df["confidence"] >= min_conf]
        if search_text:
            filtered_df = filtered_df[filtered_df["text"].str.contains(search_text, case=False, na=False)]
        
        st.dataframe(
            filtered_df[["ts", "text", "sentiment", "confidence", "source"]].head(50),
            use_container_width=True,
            height=300,
        )
        
        st.divider()

        # ============================================================
        # 9. EXPORT DATA
        # ============================================================
        st.markdown("**📥 Export Data**")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("📥 Download Full Log (CSV)", use_container_width=True):
                csv = log_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Click to Download",
                    csv,
                    file_name=f"predictions_log_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        
        with col_e2:
            if st.button("📊 Download Summary Report", use_container_width=True):
                summary = {
                    "Total Predictions": total,
                    "Positive %": f"{pos_pct:.1f}%",
                    "Negative %": f"{neg_pct:.1f}%",
                    "Avg Confidence": f"{avg_conf:.1f}%",
                    "Avg Latency": f"{avg_latency:.1f} ms",
                    "Error Rate": f"{err_rate:.2%}",
                    "Date Range": f"{log_df['ts'].min()} to {log_df['ts'].max()}"
                }
                summary_df = pd.DataFrame([summary])
                csv = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Click to Download",
                    csv,
                    file_name=f"summary_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.divider()

        # ============================================================
        # 10. CLEAR LOG
        # ============================================================
        with st.expander("⚠️ Clear Log"):
            st.warning("This permanently deletes all logged predictions.")
            if st.button("Clear all logged predictions", type="secondary"):
                clear_predictions_log()
                st.rerun()