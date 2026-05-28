# Fake News Detection using NLP and Machine Learning

---

# Project Overview

This project is a **Fake News Detection System** built using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The model predicts whether a news article or headline is **Fake News** or **Real News**.

The project uses:
- TF-IDF Vectorization
- Logistic Regression
- Streamlit Web Application

Dataset used:
- Fake and Real News Dataset from Kaggle
Dataset Link:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset


---

# Live Demo

Hosted Application Link:

```text
Add Your Hosted Link Here
```

---

# Screenshots

## Home Page

![Home Page](screenshots\home.png)

---

## Prediction Result - Fake News

![Fake News Prediction](screenshots\fake.png)

---

## Prediction Result - Real News

![Real News Prediction](screenshots\real.png)

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
- Real-time prediction system

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Pickle

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

---

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

## Logistic Regression

Logistic Regression is used as the primary machine learning model for binary text classification.

Advantages:
- Fast training
- High accuracy for text classification
- Efficient with TF-IDF features
- Lightweight and easy to deploy

---

# Project Structure

```text
FAKE_NEWS_DETECTION/
│
├── datasets/
│   ├── Fake.csv
│   └── True.csv
│
├── text_preprocessing.ipynb
├── app.py
│
├── model.pkl
├── vectorizer.pkl
│
├── requirements.txt
│
├── LICENSE
│
└── README.md
```

---

# Streamlit App

The Streamlit web application allows users to:
- Enter a news headline or article
- Predict whether the news is fake or real
- View prediction instantly

---

# How to Run the Project

## 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk streamlit
```

---

## 2. Run Notebook

Run:

```text
text_preprocessing.ipynb
```

This will:
- preprocess text
- train model
- generate:
  - model.pkl
  - vectorizer.pkl

---

## 3. Run Streamlit App

```bash
streamlit run app.py
```

---

# Deployment
- Streamlit Cloud

---

# requirements.txt

```txt
pandas
numpy
scikit-learn
nltk
streamlit
```

---

# LICENSE

```text
MIT License

Copyright (c) 2026 Anuj Wagmore & Tapaswini Shaw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
```

---

# Future Improvements

- Add prediction confidence score
- Improve UI design
- Use PassiveAggressiveClassifier
- Add BERT model
- Deploy application online
- Add live news API integration
- Multi-language fake news detection
- Dark mode UI

---
# Authors

## Anuj Wagmore
## Tapaswini Shaw

---

# Acknowledgements

- Kaggle Dataset Contributors
- Scikit-learn Documentation
- NLTK Documentation
- Streamlit Documentation

---

# Support

If you like this project, give it a ⭐ on GitHub.