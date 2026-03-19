import streamlit as st
import tensorflow as tf
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
from newspaper import Article

# Download stopwords
nltk.download("stopwords")

st.set_page_config(page_title="AI Fake Content Detector", layout="centered")

st.markdown("""
<style>

.result-card {
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.real {
    background-color:#d4edda;
    color:#155724;
}

.fake {
    background-color:#f8d7da;
    color:#721c24;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------

news_model = tf.keras.models.load_model("model/best_model.h5")
review_model = tf.keras.models.load_model("model/review_model.h5")

with open("model/news_tokenizer.pkl", "rb") as f:
    news_tokenizer = pickle.load(f)

with open("model/review_tokenizer.pkl", "rb") as f:
    review_tokenizer = pickle.load(f)

stop_words = set(stopwords.words("english"))

# -----------------------------
# CLEAN NEWS TEXT
# -----------------------------

def clean_news_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -----------------------------
# LIME PREDICTION FUNCTION
# -----------------------------

def predict_proba(texts):
    sequences = news_tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=250, padding="post")

    preds = news_model.predict(padded)

    results = []
    for p in preds:
        results.append([1 - p[0], p[0]])

    return np.array(results)

# -----------------------------
# TITLE
# -----------------------------

st.title("🧠 AI Fake Content Detection System")
st.markdown("Detect **Fake News** 📰 and **Fake Reviews** ⭐ using Deep Learning (RNN)")

# -----------------------------
# TABS
# -----------------------------

tab1, tab2 = st.tabs(["📰 Fake News Detection", "⭐ Fake Review Detection"])

# =================================================
# FAKE NEWS DETECTION
# =================================================

with tab1:

    st.subheader("📰 Fake News Detection")

    # Select input method
    input_mode = st.radio(
        "Select Input Method",
        ["Enter News Text", "Paste News URL"]
    )

    news_input = ""

    # Manual Text Input
    if input_mode == "Enter News Text":

        news_input = st.text_area("Enter News Text")

    # URL Input
    else:

        news_url = st.text_input("Paste News URL")

        if news_url != "":
            try:
                article = Article(news_url)
                article.download()
                article.parse()

                news_input = article.text

                st.subheader("Extracted Article Text")
                st.write(news_input[:500] + "...")

            except:
                st.error("Unable to extract article from URL")

    # Detect Button
if st.button("Detect Fake News"):

    if news_input.strip() == "":
        st.warning("⚠ Please enter news text or URL.")

    else:

        cleaned = clean_news_text(news_input)

        seq = news_tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=250, padding="post")

        prob = news_model.predict(padded)[0][0]
        confidence = float(prob)

        st.subheader("Prediction Result")

        if confidence >= 0.5:

            st.markdown(
                f'<div class="result-card real">✅ REAL NEWS<br>Confidence: {confidence*100:.2f}%</div>',
                unsafe_allow_html=True
            )

            st.progress(confidence)

        else:

            fake_conf = 1 - confidence

            st.markdown(
                f'<div class="result-card fake">🚨 FAKE NEWS<br>Confidence: {fake_conf*100:.2f}%</div>',
                unsafe_allow_html=True
            )

            st.progress(fake_conf)

        # -----------------------------
        # LIME EXPLANATION
        # -----------------------------

        explainer = LimeTextExplainer(class_names=["Fake", "Real"])

        exp = explainer.explain_instance(
            cleaned,
            predict_proba,
            num_features=8
        )

        st.subheader("Important Words Affecting Prediction")

        for word, weight in exp.as_list():
            st.write(f"{word} : {weight:.3f}")

# =================================================
# FAKE REVIEW DETECTION
# =================================================

with tab2:

    st.subheader("⭐ Fake Review Detection")

    review_input = st.text_area("Enter Review Text")

    if st.button("Detect Fake Review"):

        if review_input.strip() == "":
            st.warning("⚠ Please enter review text.")

        else:

            text = review_input.lower()

            seq = review_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=200)

            prob = review_model.predict(padded)[0][0]
            confidence = float(prob)

            st.subheader("Prediction Result")

            st.progress(confidence)

            if confidence >= 0.5:
                st.success("✅ This Review is REAL")
                st.write(f"Confidence: {confidence*100:.2f}%")
            else:
                st.error("🚨 This Review is FAKE")
                st.write(f"Confidence: {(1-confidence)*100:.2f}%")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")
st.caption("🧠 AI Powered Fake News & Fake Review Detection using Deep Learning (RNN) | Final Year Project")