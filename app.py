import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# NLP setup
ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

# Text preprocessing function
def text_cleaning(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    sentence = []

    for word in words:
        if word not in stop_words:
            sentence.append(ps.stem(word))

    return " ".join(sentence)

# Streamlit UI
st.title("Fake News Detection")

st.write("Enter a news headline or article")

news = st.text_area("News Text")
st.sidebar.title("About")
st.sidebar.write("Fakenews detetion app")

if st.button("Predict"):
    if news.strip() == "":
        st.error("Write a news article or headline")
    else:
        cleaned_news = text_cleaning(news)
        # st.write("cleaned news",cleaned_news)

        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)
        probability = model.predict_proba(news_vector)
        
        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")