# Fake News Detection using NLP and Machine Learning

## Project Overview

This project is a Fake News Detection System built using Natural Language Processing (NLP) and Machine Learning techniques.  
The model predicts whether a news article or headline is Fake News or Real News.

The project uses:
- TF-IDF Vectorization
- Logistic Regression
- Streamlit Web Application

Dataset used:
- Fake and Real News Dataset from Kaggle

---

# Features

- Text preprocessing using NLP
- Stopword removal
- Stemming using Porter Stemmer
- TF-IDF vectorization
- Logistic Regression model
- Fake/Real news prediction
- Streamlit web application
- User-friendly interface

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit

---
# Clone Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
```

Move into project folder:

```bash
cd fake-news-detection
```

---

# Create Virtual Environment

```bash
python -m venv .env
```

Activate virtual environment:

## Windows

```bash
.env\Scripts\activate
```

## Linux / Mac

```bash
source .env/bin/activate
```

---

# Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk streamlit
```

# Dataset

Dataset Link:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset contains:
- Fake.csv
- True.csv

---

# NLP Preprocessing Steps

The following preprocessing techniques were applied:

1. Lowercase conversion
2. Removal of special characters
3. Tokenization
4. Stopword removal
5. Stemming using PorterStemmer

Example:

Before preprocessing:

```text
Donald Trump Sends Out Embarrassing New Year Eve Message
```

After preprocessing:

```text
donald trump send embarrass new year eve messag
```

---

# Machine Learning Workflow

```text
Dataset
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Prediction
```

---

# Model Used

- Logistic Regression

---

# Project Structure

```text
FAKE_NEWS_DETECTION/
│
├── datasets/
│
├── text_preprocessing.ipynb
├── app.py
│
├── model.pkl
├── vectorizer.pkl
│
└── README.md
```

---

# Streamlit App

The Streamlit web application allows users to:
- Enter a news headline or article
- Predict whether the news is fake or real

---

# How to Run the Project

## 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk streamlit
```

---

## 2. Run Notebook

Run:
- `text_preprocessing.ipynb`

This will:
- preprocess text
- train model
- generate:
  - `model.pkl`
  - `vectorizer.pkl`

---

## 3. Run Streamlit App

```bash
streamlit run app.py
```

---

# Future Improvements

- Add prediction confidence score
- Improve UI design
- Use PassiveAggressiveClassifier
- Add BERT model
- Deploy application online
- Add live news API integration

---

# Author

Anuj Wagmore & Tapaswini Shaw
