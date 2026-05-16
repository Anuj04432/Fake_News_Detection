
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# Downloading stopwords
nltk.download('stopwords')

# Loading dattasets
real = pd.read_csv(r"C:\Users\Anuj Kumar\Desktop\Fake_News_Detection\datasets\True.csv")
fake = pd.read_csv(r"C:\Users\Anuj Kumar\Desktop\Fake_News_Detection\datasets\Fake.csv")

df1 = fake.copy()
df2 = real.copy()

# Applying stemming for all content
ps = PorterStemmer()

def text_cleaning(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z]',' ',text)
    stop_words = set(stopwords.words('english'))

    words = text.split()
    sentence = []
    for word in  words:
        if word not in stop_words:
            sentence.append(ps.stem(word))

    return " ".join(sentence)

# Creating labels for fake and real news

df1["label"] = 0
df2["label"] = 1

df = pd.concat([df1,df2],axis=0)

# shuffling rows
df = df.sample(frac=1).reset_index(drop=True)
df['content'] = df['title']+" "+df["text"]
df["content"]=df["content"].apply(text_cleaning)

#Train Test Split

X = df["content"]
y  = df["label"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)
print(y_test.shape)

# converting text into vectors
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

print(X_train.shape,X_test.shape)

# Model Training
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


# Input values and Prediction
news = "Narendra modi is the chief minister of China"
cleaned_news = text_cleaning(news)

news_vector = tfidf.transform([cleaned_news])

prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")




