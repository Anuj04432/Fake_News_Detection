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

st.markdown("""
<style>
    textarea{ 
            background-color:white !important;
            color:white !important;
            font-family: 'Segoe UI', sans-serif !important;
            border:2px solid !important;
            border-radius:12px !important;
            padding:14px !important;
            font-size:px !important;
            transition:border-color 0.3s ease box-shadow 0.3s ease !important;
            resize: vertical !important;
            }

    textarea:focus {
    border-color: #cba6f7 !important;
    box-shadow: 0 4px 20px rgba(203, 166, 247, 0.4) !important;
    outline: none !important;
}
     textarea::placeholder {
    color: #6c7086 !important;
    font-style: italic;
}
            </style>
""",unsafe_allow_html=True)

news = st.text_area("Headline||News article", placeholder="Type something beautiful...")
st.sidebar.title("📰 About This Project")

st.sidebar.info(
    """
    ## Fake News Detection System

    This application uses:

    ✅ Natural Language Processing (NLP)  
    ✅ TF-IDF Vectorization  
    ✅ Logistic Regression Model  
    ✅ Streamlit Web Application  

    Enter a news headline or article to check whether it is:

    ✔ Real News  
    ✖ Fake News
    """
)
with st.sidebar:
    col1,col2,col3 = st.columns(3)

    with col1:
        st.link_button("</>Github","https://github.com/Anuj04432")

    with col2:
        st.link_button("ℹ️Linkedin","https://www.linkedin.com/in/anuj-wagmore-874a883a7/")
    with col3:
        st.link_button("🌐Portfolio","https://anujwagmore.netlify.app/")
    # with col4:
    #     st.link_button("</>Github","https://github.com/Anuj04432")



if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("⚠ Please write a news article or headline")

    else:

        cleaned_news = text_cleaning(news)

        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)

        probability = model.predict_proba(news_vector)

        confidence = max(probability[0]) * 100

        st.subheader("Prediction Result")

        if prediction[0] == 0:

            st.error("🚨 Fake News Detected")

            st.write(f"Confidence Score: {confidence:.2f}%")

        else:

            st.success("✅ Real News")

            st.write(f"Confidence Score: {confidence:.2f}%")

        # with st.expander("View Cleaned Text"):
        #     st.write(cleaned_news)