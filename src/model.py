import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# load data
df = pd.read_csv("data/judgments.csv")

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['judgment_text'].apply(clean_text)

# vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

# target
y = df['outcome']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = LogisticRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluation
print(classification_report(y_test, y_pred))

# Bias analysis: Gender vs Outcome

print("\n--- Bias Analysis (Gender vs Outcome) ---")
bias_gender = pd.crosstab(df['gender'], df['outcome'], normalize='index')
print(bias_gender)

# Bias analysis: Region vs Outcome

print("\n--- Bias Analysis (Region vs Outcome) ---")
bias_region = pd.crosstab(df['region'], df['outcome'], normalize='index')
print(bias_region)