import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os

# Tell nltk to use local bundled data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))


ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Spam Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .title {
        text-align: center;
        color: #4a4a4a;
    }
    .result-box {
        background-color: #e0f7fa;
        border: 1px solid #00acc1;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #00796b;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">📧 Email/SMS Spam Detection</h1>', unsafe_allow_html=True)

# Layout
with st.container():
    input_sms = st.text_area("Enter your message below 👇", height=150)

    if st.button('🔍 Predict'):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter a message before clicking Predict.")
        else:
            # Preprocess
            transformed_sms = transform_text(input_sms)
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # Predict
            result = model.predict(vector_input)[0]

            # Output
            if result == 1:
                st.markdown('<div class="result-box">🚫 Spam</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box">✅ Not Spam</div>', unsafe_allow_html=True)
