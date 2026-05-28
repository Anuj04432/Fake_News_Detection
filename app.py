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
st.markdown(
    "<h1 style='font-size:70px; text-align:center; color:Green;'>Fake News Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:30px;'>📰Enter a news headline or article</p>",
    unsafe_allow_html=True
)
st.markdown("""
<style>

/* Label color */
label {
    color:  !important;
    font-size: 40px !important;
    font-weight: bold;
}

/* Textarea styling */
textarea {
    font-size: 15px !important;
    color: black !important;
    background-color: #FCFFC9 !important;
    caret-color: black !important; 
    border-radius: 10px !important;
    border: 2px solid #4CAF50 !important;
}

/* Placeholder color */
textarea::placeholder {
    color: gray !important;
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)
# Text area
news = st.text_area(label="Enter News Headline or Arrticle",
    placeholder="Type or paste the news article here...",
    height=150
)

# About the project
st.sidebar.title("📰 About This Project")

st.sidebar.info(
    """
    ## Fake News Detection System

    This application uses:

    ✅ Natural Language Processing (NLP)  
    ✅ TF-IDF Vectorization  
    ✅ Logistic Regression Model  
    ✅ Streamlit Web Application  
    ✅ Text Preprocessing & Cleaning  
    ✅ Stopword Removal & Stemming  

    ### Features
    🔹 Detects Fake and Real News  
    🔹 Instant Prediction Results  
    🔹 User-Friendly Interface  
    🔹 Machine Learning Based Detection  

    Enter a news headline or article to check whether it is:

    ✔ Real News  
    ✖ Fake News

    ---
    Developed by:
    👨‍💻 Anuj & Tapaswini Shaw
    """
)
# Connection Links
# with st.sidebar:
#     col1,col2,col3 = st.columns(3)

#     with col1:
#         st.link_button("</>Github","https://github.com/Anuj04432")

#     with col2:
#         st.link_button("ℹ️Linkedin","https://www.linkedin.com/in/anuj-wagmore-874a883a7/")
#     with col3:
#         st.link_button("🌐Portfolio","https://anujwagmore.netlify.app/")

# Prediction button
if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("⚠ Please enter a news headline or article")

    else:

        # Text preprocessing
        cleaned_news = text_cleaning(news)

        # Convert text into vectors
        news_vector = vectorizer.transform([cleaned_news])

        # Prediction
        prediction = model.predict(news_vector)

        # Prediction probability
        probability = model.predict_proba(news_vector)

        # Confidence score
        confidence = max(probability[0]) * 100

        st.subheader("📌 Prediction Result")

        if prediction[0] == 0:

            st.error("🚨 Fake News Detected")
            st.write(f"### Confidence Score: {confidence:.2f}%")

        else:

            st.success("✅ Real News")
            st.write(f"### Confidence Score: {confidence:.2f}%")

        # Additional Information
        st.markdown("---")
        st.info(
            """
            ⚠ This prediction is generated using a Machine Learning model
            and may not always be 100% accurate.

            Always verify important news from trusted sources.
            """
        )
