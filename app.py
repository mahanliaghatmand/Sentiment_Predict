import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Twitter Sentiment Analysis")
st.markdown("---")

@st.cache_resource
def load_model_and_vectorizer():
    # مسیرهای اصلاح شده برای Streamlit Cloud
    # فایل app.py در ریشه پروژه است، پس مستقیماً به پوشه‌ها اشاره می‌کنیم
    data_path = os.path.join(os.path.dirname(__file__), "data", "twitter_training_clean_binary.csv")
    df = pd.read_csv(data_path)
    
    X = df['clean_text'].values
    y = df['sentiment'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Rebuild vectorizer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=20000,
        output_sequence_length=40,
        standardize='lower_and_strip_punctuation'
    )
    vectorizer.adapt(X_train)
    
    # Rebuild model architecture
    def build_model():
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = vectorizer(inputs)
        
        embedding_dim = 128
        x = tf.keras.layers.Embedding(
            input_dim=20000,
            output_dim=embedding_dim,
            mask_zero=True
        )(x)
        
        positions = tf.range(start=0, limit=40, delta=1)
        pos_embedding = tf.keras.layers.Embedding(
            input_dim=40,
            output_dim=embedding_dim
        )(positions)
        x = x + pos_embedding
        
        def transformer_block(x):
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=embedding_dim // 4
            )(x, x)
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization()(x)
            
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128)
            ])
            ffn_output = ffn(x)
            x = tf.keras.layers.Add()([x, ffn_output])
            x = tf.keras.layers.LayerNormalization()(x)
            return x
        
        x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model
    
    model = build_model()
    
    # مسیر مدل - اصلاح شده برای Streamlit Cloud
    weights_path = os.path.join(os.path.dirname(__file__), "model", "model_scratch_sentiment.keras")
    model.load_weights(weights_path)
    
    return model, vectorizer

try:
    model, vectorizer = load_model_and_vectorizer()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    - **Architecture:** Transformer Encoder (Scratch)
    - **Task:** Binary Sentiment (Positive/Negative)
    - **Sequence Length:** 40 tokens
    - **Embedding Dimension:** 128
    - **Attention Heads:** 4
    - **Vocab Size:** 20,000
    """)
    
    st.header("🧪 Test Examples")
    test_texts = [
        "I love this product! It's amazing",
        "This is terrible, I hate it",
        "The movie was okay, not great",
        "Best day ever! Feeling so happy"
    ]
    
    for text in test_texts:
        pred = model.predict([text], verbose=0)[0][0]
        sentiment = "😊 Positive" if pred > 0.5 else "😠 Negative"
        confidence = pred if pred > 0.5 else 1 - pred
        st.write(f"**{text}**")
        st.write(f"→ {sentiment} (confidence: {confidence:.2%})")
        st.divider()

# Main interface
tab1, tab2 = st.tabs(["📝 Single Tweet", "📤 Batch Analysis"])

with tab1:
    st.header("Analyze a Single Tweet")
    user_input = st.text_area(
        "Enter your tweet text:",
        placeholder="Type your tweet here...",
        height=100
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    prediction = model.predict([user_input], verbose=0)[0][0]
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    sentiment = "😊 Positive" if prediction > 0.5 else "😠 Negative"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col3:
                        st.metric("Raw Score", f"{prediction:.4f}")
                    
                    st.progress(float(confidence))
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Batch Analysis via CSV Upload")
    st.markdown("""
    Upload a CSV file with a column named **'text'** containing the tweets to analyze.
    The results will include the original text, predicted sentiment, and confidence score.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("❌ CSV must contain a 'text' column")
            else:
                with st.spinner(f"Analyzing {len(df)} tweets..."):
                    predictions = []
                    batch_size = 32
                    for i in range(0, len(df), batch_size):
                        batch = df['text'].iloc[i:i+batch_size].tolist()
                        batch_preds = model.predict(batch, verbose=0)
                        predictions.extend([p[0] for p in batch_preds])
                    
                    df['sentiment'] = ['😊 Positive' if p > 0.5 else '😠 Negative' for p in predictions]
                    df['confidence'] = [p if p > 0.5 else 1-p for p in predictions]
                    df['raw_score'] = predictions
                    
                    st.success(f"✅ Analysis complete! Processed {len(df)} tweets.")
                    
                    st.subheader("Preview Results")
                    st.dataframe(df.head(10))
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Keras, and Streamlit")