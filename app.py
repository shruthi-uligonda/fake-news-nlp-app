import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Fake News Detection | NLP",
    page_icon="📰",
    layout="centered"
)

# -------------------------------
# Load model & vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# NLP preprocessing
# -------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# Header section
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>📰 Fake News Detection using NLP</h1>
    <p style='text-align: center; font-size: 18px;'>
    Analyze news articles and predict whether they are <b>REAL</b> or <b>FAKE</b>
    using Machine Learning and Natural Language Processing.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Input section
# -------------------------------
st.subheader("📄 Enter News Content")

news_text = st.text_area(
    label="Paste the full news article text below",
    height=220,
    placeholder="Copy and paste news text from any website (BBC, TOI, The Hindu, etc.)"
)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔍 Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text before clicking Analyze.")
    else:
        clean_text = preprocess(news_text)
        vector = vectorizer.transform([clean_text])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0].max() * 100

        st.markdown("---")
        st.subheader("📊 Analysis Result")

        if prediction == "REAL":
            st.success(f"✅ **Prediction: REAL NEWS**")
        else:
            st.error(f"❌ **Prediction: FAKE NEWS**")

        st.info(f"📈 **Confidence Level:** {confidence:.2f}%")

# -------------------------------
# Footer / explanation
# -------------------------------
st.markdown("---")
st.markdown(
    """
    ### ℹ️ About this system
    - This application uses **TF-IDF + Naive Bayes** for text classification  
    - It performs **content-based analysis**, not real-time fact verification  
    - Accuracy depends on training data and writing patterns  

    ⚠️ *This tool is for educational and research purposes only.*
    """
)