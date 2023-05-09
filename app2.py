import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Set page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add page background color and image
page_bg = """
<style>
body {
background-color: #0a1931;
background-image: url("https://images.unsplash.com/photo-1489862460826-944d06a3c173");
background-size: cover;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Add CSS style for title and button
st.markdown(
    """
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 50px;
    }
    .small-font {
        font-size:25px !important;
        color: white;
        margin-top: 50px;
        margin-bottom: 50px;
    }
    .predict-button {
        background-color: #cc3300 !important;
        color: white !important;
        font-weight: bold;
        margin-top: 20px;
        margin-left: 10px;
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add title
st.markdown('<p class="big-font">Email/SMS Spam Classifier</p>', unsafe_allow_html=True)

# Add text area and predict button
input_sms = st.text_area("Enter the message", height=150)

col1, col2 = st.beta_columns(2)
if col2.button('Predict', key='predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        col1.markdown('<p class="small-font">Prediction: <b>Spam</b></p>', unsafe_allow_html=True)
    else:
        col1.markdown('<p class="small-font">Prediction: <b>Not Spam</b></p>', unsafe_allow_html=True)

