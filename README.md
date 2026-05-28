# Fake News Detection using NLP and Machine Learning

## Project Overview
This project is a **Fake News Detection System** built using **Natural Language Processing (NLP)** and **Machine Learning**.  
The model predicts whether a news headline or article is **Fake News** or **Real News**.

The system uses:
- TF-IDF Vectorization
- Logistic Regression
- Streamlit Web Application

---

## Dataset
The project uses the **Fake and Real News Dataset** from Kaggle.

### Dataset Files
- `Fake.csv` – Contains fake news
- `True.csv` – Contains real news

### Dataset Link
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Features / Functionality

### NLP Preprocessing
- Lowercase conversion
- Removal of special characters
- Tokenization
- Stopword removal
- Stemming using PorterStemmer

### Machine Learning
- TF-IDF vectorization
- Logistic Regression model
- Fake/Real news prediction
- Confidence score prediction

### Streamlit Web Application
- User-friendly interface
- Sidebar project description
- GitHub, LinkedIn, and Portfolio links
- News headline/article input area

---

## Example of Text Preprocessing

### Before preprocessing
Donald Trump Sends Out Embarrassing New Year Eve Message

### After preprocessing
donald trump send embarrass new year eve messag

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit

---

## Project Structure

FAKE_NEWS_DETECTION/
│
├── datasets/
│   ├── Fake.csv
│   └── True.csv
│
├── text_preprocessing.ipynb
├── app.py
├── model.pkl
├── vectorizer.pkl
└── README.md

---

## Machine Learning Workflow

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

---

## Model Used
- Logistic Regression

---

## How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Create Virtual Environment
```bash
python -m venv .env
```

### 3. Activate Virtual Environment

#### Windows
```bash
.env\Scripts\activate
```

#### Linux / Mac
```bash
source .env/bin/activate
```

### 4. Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk streamlit
```

### 5. Run Notebook
Run:
```bash
text_preprocessing.ipynb
```

This will:
- preprocess text
- train model
- generate:
  - model.pkl
  - vectorizer.pkl

### 6. Run Streamlit App
```bash
streamlit run app.py
```

---

## Streamlit App Features
- Enter a news headline or article
- Predict whether the news is fake or real
- Display prediction confidence score

---

## Future Improvements
- Add PassiveAggressiveClassifier
- Add BERT model
- Improve UI design
- Deploy application online
- Add live news API integration
