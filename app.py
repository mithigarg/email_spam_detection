import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# load the HTML template
def get_template(template_name):
    with open(f"templates/{template_name}.html") as f:
        template = f.read()
    return template

# load the CSS stylesheet
def get_css():
    with open("static/style.css") as f:
        css = f.read()
    return css


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


# main app function
def main():
    st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📨")

    # load the CSS stylesheet
    css = get_css()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.title("Email/SMS Spam Classifier")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        # preprocess the input text
        transformed_sms = transform_text(input_sms)
        # vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        # predict the result
        result = model.predict(vector_input)[0]
        # display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

if __name__ == "__main__":
    main()
